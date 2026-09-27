/****************************************************************************
 *
 * MODULE:       i.sar.coregistration
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Public interface of libsarcoreg, the compute library of
 *               i.sar.coregistration, loaded from Python through ctypes.
 * COPYRIGHT:    (C) 2026 by Yann Chemin and the GRASS Development Team
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 *****************************************************************************/

#ifndef SARCOREG_H
#define SARCOREG_H

#include <stdint.h>

/* One reference segment (rows of one burst) to coregister. Mirrored by
 * the ctypes structure SarcoregJob of i.sar.coregistration.py: keep both
 * in the same order. */
struct sarcoreg_job {
    int rows, cols;             /* Output pixels. */
    double line0;               /* Reference burst line of output row 0. */
    int ngl, ngs;               /* Geometry nodes in azimuth and range. */
    const double *grid_lines;   /* ngl reference burst lines, increasing. */
    const double *grid_samples; /* ngs reference range samples, increasing. */
    const double *nodes;        /* NODE_COUNT arrays of ngl * ngs values. */
    int has_dem;
    double h_lo, h_hi; /* Node heights of the _LO and _HI arrays. */
    const float *dem;  /* dem_rows * dem_cols, NaN for no-data. */
    int dem_rows, dem_cols;
    double dem_lat0, dem_lon0; /* Centre of the first DEM cell. */
    double dem_dlat, dem_dlon; /* DEM cell size (dem_dlat < 0). */
    int nseg, sec_cols;        /* Secondary segments and range samples. */
    const float *sec;          /* Interleaved re, im, NaN for null. */
    const int64_t *segidx;     /* Per segment: offset (pixels), rows. */
    const double *seginfo;     /* Per segment: u of row 0, burst line. */
    const double *doppler;     /* Per segment: kt, tref, fdc per sample. */
    double dt;                 /* Secondary azimuth time interval. */
    double shift;              /* Azimuth shift added to u (ESD). */
    double range_shift;        /* Range shift added to x (samples). */
    int kernel_type, taps;     /* SARCOREG_BILINEAR/BICUBIC/SINC. */
    float *out;                /* rows * cols interleaved re, im. */
    float *aux;                /* AUX_COUNT arrays of rows * cols. */
    int want_aux;
};

/* Select the processing device: "host" (C, OpenMP), "auto" (first OpenCL
 * GPU, then OpenCL CPU, then host), "gpu" or "cpu" (OpenCL only).
 * platform, when not NULL or empty, restricts OpenCL to platforms whose
 * name contains it (case-insensitive). nthreads limits the OpenMP threads
 * of the host path (0: default). Returns 0 on success with a description
 * of the device in msg, nonzero with the error in msg. */
int sarcoreg_init(const char *device, const char *platform, int nthreads,
                  char *msg, int msglen);

/* Coregister one segment. Returns 0 on success, nonzero with the error
 * in msg. */
int sarcoreg_run(const struct sarcoreg_job *job, char *msg, int msglen);

void sarcoreg_finish(void);

#endif /* SARCOREG_H */
