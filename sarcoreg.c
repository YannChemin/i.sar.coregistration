/****************************************************************************
 *
 * MODULE:       i.sar.coregistration
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      libsarcoreg: runs the per-pixel coregistration of
 *               sarcoreg_core.h on the host (OpenMP) or on an OpenCL
 *               device. Errors are returned to the caller, never fatal.
 * COPYRIGHT:    (C) 2026 by Yann Chemin and the GRASS Development Team
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 *****************************************************************************/

/* Target the OpenCL 1.2 API, the common denominator of PoCL, Mesa and the
 * discrete GPU vendor ICDs. */
#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <CL/cl.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "sarcoreg.h"
#include "sarcoreg_core.h"

/* sarcoreg_core.h and sarcoreg_kernels.cl as one string literal, generated
 * by the Makefile. */
static const char *kernel_source =
#include "sarcoreg_cl.h"
    ;

static struct {
    int ready;
    int use_ocl;
    int nthreads;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_device_id device;
    cl_ulong max_alloc;
} state;

static void set_msg(char *msg, int msglen, const char *fmt, ...)
{
    va_list ap;

    if (!msg || msglen <= 0)
        return;
    va_start(ap, fmt);
    vsnprintf(msg, (size_t)msglen, fmt, ap);
    va_end(ap);
}

static int contains_nocase(const char *hay, const char *needle)
{
    size_t n = strlen(needle), i, j;

    if (n == 0)
        return 1;
    for (i = 0; hay[i]; i++) {
        for (j = 0; j < n && hay[i + j]; j++)
            if (tolower((unsigned char)hay[i + j]) !=
                tolower((unsigned char)needle[j]))
                break;
        if (j == n)
            return 1;
    }
    return 0;
}

static void release_ocl(void)
{
    if (state.kernel)
        clReleaseKernel(state.kernel);
    if (state.program)
        clReleaseProgram(state.program);
    if (state.queue)
        clReleaseCommandQueue(state.queue);
    if (state.context)
        clReleaseContext(state.context);
    state.kernel = NULL;
    state.program = NULL;
    state.queue = NULL;
    state.context = NULL;
}

/* Build the kernels on one device; 0 on success. */
static int try_device(cl_platform_id platform, cl_device_id device, char *msg,
                      int msglen)
{
    cl_int err;
    size_t len = strlen(kernel_source);
    char name[256], pname[256], ext[4096];
    cl_context_properties props[3] = {CL_CONTEXT_PLATFORM,
                                      (cl_context_properties)platform, 0};

    clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, sizeof(ext), ext, NULL);
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, NULL);
    clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(pname), pname, NULL);
    if (!strstr(ext, "cl_khr_fp64")) {
        set_msg(msg, msglen,
                "OpenCL device '%s' has no double precision (cl_khr_fp64)",
                name);
        return 1;
    }
    state.context = clCreateContext(props, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        set_msg(msg, msglen, "clCreateContext failed on '%s' (%d)", name, err);
        return 1;
    }
    state.queue = clCreateCommandQueue(state.context, device, 0, &err);
    if (err != CL_SUCCESS) {
        set_msg(msg, msglen, "clCreateCommandQueue failed on '%s' (%d)", name,
                err);
        release_ocl();
        return 1;
    }
    state.program =
        clCreateProgramWithSource(state.context, 1, &kernel_source, &len, &err);
    if (err == CL_SUCCESS)
        err = clBuildProgram(state.program, 1, &device, "", NULL, NULL);
    if (err != CL_SUCCESS) {
        char log[4096] = "";

        if (state.program)
            clGetProgramBuildInfo(state.program, device, CL_PROGRAM_BUILD_LOG,
                                  sizeof(log), log, NULL);
        set_msg(msg, msglen, "OpenCL build failed on '%s' (%d): %s", name, err,
                log);
        release_ocl();
        return 1;
    }
    state.kernel = clCreateKernel(state.program, "coregister", &err);
    if (err != CL_SUCCESS) {
        set_msg(msg, msglen, "clCreateKernel failed on '%s' (%d)", name, err);
        release_ocl();
        return 1;
    }
    clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                    sizeof(state.max_alloc), &state.max_alloc, NULL);
    state.device = device;
    state.use_ocl = 1;
    set_msg(msg, msglen, "OpenCL device '%s' (platform '%s')", name, pname);
    return 0;
}

/* Try the devices of one type on the matching platforms, in order. */
static int try_type(cl_device_type type, const char *platform_opt, char *msg,
                    int msglen)
{
    cl_platform_id platforms[16];
    cl_device_id devices[16];
    cl_uint np = 0, nd, p, d;
    char pname[256];

    if (clGetPlatformIDs(16, platforms, &np) != CL_SUCCESS || np == 0) {
        set_msg(msg, msglen, "no OpenCL platform found");
        return 1;
    }
    set_msg(msg, msglen, "no matching OpenCL device found");
    for (p = 0; p < np; p++) {
        clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME, sizeof(pname), pname,
                          NULL);
        if (platform_opt && !contains_nocase(pname, platform_opt))
            continue;
        if (clGetDeviceIDs(platforms[p], type, 16, devices, &nd) != CL_SUCCESS)
            continue;
        for (d = 0; d < nd; d++)
            if (try_device(platforms[p], devices[d], msg, msglen) == 0)
                return 0;
    }
    return 1;
}

int sarcoreg_init(const char *device, const char *platform, int nthreads,
                  char *msg, int msglen)
{
    const char *plat = (platform && *platform) ? platform : NULL;

    sarcoreg_finish();
    state.nthreads = nthreads;
    state.ready = 1;
    if (strcmp(device, "host") == 0) {
#ifdef _OPENMP
        set_msg(msg, msglen, "host CPU, %d OpenMP threads",
                nthreads > 0 ? nthreads : omp_get_max_threads());
#else
        set_msg(msg, msglen, "host CPU, single thread");
#endif
        return 0;
    }
    if (strcmp(device, "gpu") == 0 || strcmp(device, "auto") == 0) {
        if (try_type(CL_DEVICE_TYPE_GPU, plat, msg, msglen) == 0)
            return 0;
        if (strcmp(device, "gpu") == 0) {
            state.ready = 0;
            return 1;
        }
    }
    if (try_type(CL_DEVICE_TYPE_CPU, plat, msg, msglen) == 0)
        return 0;
    if (strcmp(device, "auto") == 0) {
        /* Fall back to OpenMP, keeping why OpenCL was not used. */
        char reason[1024] = "";
        size_t n;

        if (msg && msglen > 0)
            snprintf(reason, sizeof(reason), "%s", msg);
        sarcoreg_init("host", NULL, nthreads, msg, msglen);
        n = msg ? strlen(msg) : 0;
        if (msg && (int)n < msglen)
            snprintf(msg + n, (size_t)msglen - n, " (no OpenCL: %s)", reason);
        return 0;
    }
    state.ready = 0;
    return 1;
}

void sarcoreg_finish(void)
{
    release_ocl();
    state.use_ocl = 0;
    state.ready = 0;
}

static void run_host(const struct sarcoreg_job *j)
{
    long npix = (long)j->rows * j->cols;
    int r;

#ifdef _OPENMP
    if (state.nthreads > 0)
        omp_set_num_threads(state.nthreads);
#endif
#pragma omp parallel for schedule(dynamic, 4)
    for (r = 0; r < j->rows; r++) {
        int c;

        for (c = 0; c < j->cols; c++)
            sarcoreg_pixel(
                r, c, j->cols, j->line0, j->ngl, j->ngs, j->grid_lines,
                j->grid_samples, j->nodes, j->has_dem, j->h_lo, j->h_hi, j->dem,
                j->dem_rows, j->dem_cols, j->dem_lat0, j->dem_lon0, j->dem_dlat,
                j->dem_dlon, j->nseg, j->sec_cols, j->sec, j->segidx,
                j->seginfo, j->doppler, j->dt, j->shift, j->range_shift,
                j->kernel_type, j->taps, j->out, j->aux, j->want_aux, npix);
    }
}

/* Device buffer from host memory; a one-element dummy when n is 0. */
static cl_mem upload(const void *src, size_t n, cl_mem_flags flags, cl_int *err)
{
    static const double dummy = 0.0;

    if (n == 0 || !src)
        return clCreateBuffer(state.context,
                              CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              sizeof(dummy), (void *)&dummy, err);
    return clCreateBuffer(state.context, flags | CL_MEM_COPY_HOST_PTR, n,
                          (void *)src, err);
}

static int run_ocl(const struct sarcoreg_job *j, char *msg, int msglen)
{
    enum {
        B_GL,
        B_GS,
        B_NODES,
        B_DEM,
        B_SEC,
        B_SEGIDX,
        B_SEGINFO,
        B_DOP,
        B_OUT,
        B_AUX,
        B_COUNT
    };
    cl_mem buf[B_COUNT] = {0};
    size_t npix = (size_t)j->rows * j->cols, nodes = (size_t)j->ngl * j->ngs;
    size_t sec_n = 0, gws[2], lws[2] = {16, 16};
    cl_int err = CL_SUCCESS, e;
    int k, arg = 0, ret = 1;
    int64_t s;

    /* Secondary pixels up to the end of the last segment. */
    for (k = 0; k < j->nseg; k++) {
        s = j->segidx[2 * k] + j->segidx[2 * k + 1] * j->sec_cols;
        if ((size_t)s > sec_n)
            sec_n = (size_t)s;
    }
    sec_n *= 2 * sizeof(float);
    if (sec_n > state.max_alloc || npix * 2 * sizeof(float) > state.max_alloc) {
        set_msg(msg, msglen,
                "secondary data (%.0f MB) exceed the OpenCL device allocation "
                "limit (%.0f MB); use device=host",
                sec_n / 1048576.0, state.max_alloc / 1048576.0);
        return 1;
    }

    buf[B_GL] =
        upload(j->grid_lines, j->ngl * sizeof(double), CL_MEM_READ_ONLY, &e);
    err |= e;
    buf[B_GS] =
        upload(j->grid_samples, j->ngs * sizeof(double), CL_MEM_READ_ONLY, &e);
    err |= e;
    buf[B_NODES] = upload(j->nodes, NODE_COUNT * nodes * sizeof(double),
                          CL_MEM_READ_ONLY, &e);
    err |= e;
    buf[B_DEM] = upload(
        j->has_dem ? j->dem : NULL,
        j->has_dem ? (size_t)j->dem_rows * j->dem_cols * sizeof(float) : 0,
        CL_MEM_READ_ONLY, &e);
    err |= e;
    buf[B_SEC] = upload(j->sec, sec_n, CL_MEM_READ_ONLY, &e);
    err |= e;
    buf[B_SEGIDX] =
        upload(j->segidx, 2 * j->nseg * sizeof(int64_t), CL_MEM_READ_ONLY, &e);
    err |= e;
    buf[B_SEGINFO] =
        upload(j->seginfo, 2 * j->nseg * sizeof(double), CL_MEM_READ_ONLY, &e);
    err |= e;
    buf[B_DOP] =
        upload(j->doppler, 3 * (size_t)j->nseg * j->sec_cols * sizeof(double),
               CL_MEM_READ_ONLY, &e);
    err |= e;
    buf[B_OUT] = clCreateBuffer(state.context, CL_MEM_WRITE_ONLY,
                                npix * 2 * sizeof(float), NULL, &e);
    err |= e;
    buf[B_AUX] = clCreateBuffer(state.context, CL_MEM_WRITE_ONLY,
                                j->want_aux ? AUX_COUNT * npix * sizeof(float)
                                            : sizeof(float),
                                NULL, &e);
    err |= e;
    if (err != CL_SUCCESS) {
        set_msg(msg, msglen, "OpenCL buffer allocation failed");
        goto done;
    }

#define SETARG(v) err |= clSetKernelArg(state.kernel, arg++, sizeof(v), &(v))
    SETARG(j->rows);
    SETARG(j->cols);
    SETARG(j->line0);
    SETARG(j->ngl);
    SETARG(j->ngs);
    SETARG(buf[B_GL]);
    SETARG(buf[B_GS]);
    SETARG(buf[B_NODES]);
    SETARG(j->has_dem);
    SETARG(j->h_lo);
    SETARG(j->h_hi);
    SETARG(buf[B_DEM]);
    SETARG(j->dem_rows);
    SETARG(j->dem_cols);
    SETARG(j->dem_lat0);
    SETARG(j->dem_lon0);
    SETARG(j->dem_dlat);
    SETARG(j->dem_dlon);
    SETARG(j->nseg);
    SETARG(j->sec_cols);
    SETARG(buf[B_SEC]);
    SETARG(buf[B_SEGIDX]);
    SETARG(buf[B_SEGINFO]);
    SETARG(buf[B_DOP]);
    SETARG(j->dt);
    SETARG(j->shift);
    SETARG(j->range_shift);
    SETARG(j->kernel_type);
    SETARG(j->taps);
    SETARG(buf[B_OUT]);
    SETARG(buf[B_AUX]);
    SETARG(j->want_aux);
#undef SETARG
    if (err != CL_SUCCESS) {
        set_msg(msg, msglen, "clSetKernelArg failed");
        goto done;
    }
    gws[0] = (size_t)(j->cols + lws[0] - 1) / lws[0] * lws[0];
    gws[1] = (size_t)(j->rows + lws[1] - 1) / lws[1] * lws[1];
    err = clEnqueueNDRangeKernel(state.queue, state.kernel, 2, NULL, gws, lws,
                                 0, NULL, NULL);
    if (err == CL_INVALID_WORK_GROUP_SIZE)
        err = clEnqueueNDRangeKernel(state.queue, state.kernel, 2, NULL, gws,
                                     NULL, 0, NULL, NULL);
    if (err == CL_SUCCESS)
        err = clEnqueueReadBuffer(state.queue, buf[B_OUT], CL_TRUE, 0,
                                  npix * 2 * sizeof(float), j->out, 0, NULL,
                                  NULL);
    if (err == CL_SUCCESS && j->want_aux)
        err = clEnqueueReadBuffer(state.queue, buf[B_AUX], CL_TRUE, 0,
                                  AUX_COUNT * npix * sizeof(float), j->aux, 0,
                                  NULL, NULL);
    if (err != CL_SUCCESS) {
        set_msg(msg, msglen, "OpenCL kernel execution failed (%d)", err);
        goto done;
    }
    ret = 0;

done:
    for (k = 0; k < B_COUNT; k++)
        if (buf[k])
            clReleaseMemObject(buf[k]);
    return ret;
}

int sarcoreg_run(const struct sarcoreg_job *job, char *msg, int msglen)
{
    if (!state.ready) {
        set_msg(msg, msglen, "sarcoreg_init() was not called");
        return 1;
    }
    if (job->taps < 2 || job->taps > SARCOREG_MAX_TAPS || job->ngl < 2 ||
        job->ngs < 2 || job->nseg < 1) {
        set_msg(msg, msglen, "invalid job (taps %d, nodes %dx%d, segments %d)",
                job->taps, job->ngl, job->ngs, job->nseg);
        return 1;
    }
    if (state.use_ocl)
        return run_ocl(job, msg, msglen);
    run_host(job);
    return 0;
}
