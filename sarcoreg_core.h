/****************************************************************************
 *
 * MODULE:       i.sar.coregistration
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Per-pixel coregistration of a secondary TOPS SLC burst onto
 *               the reference geometry: geometry interpolation, DEM height
 *               solution, deramping, interpolation and rerampping.
 *               This file is compiled both as C99 (host, OpenMP) and as
 *               OpenCL C (device), so that both paths run the same code.
 * COPYRIGHT:    (C) 2026 by Yann Chemin and the GRASS Development Team
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 *****************************************************************************/

#ifndef SARCOREG_CORE_H
#define SARCOREG_CORE_H

#ifdef __OPENCL_VERSION__
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define GLOBAL __global
/* Plain functions: a C99 "inline" definition emits no symbol, and drivers
 * that decline to inline it (Mesa Clover) then fail to link the kernel. */
#define INLINE
typedef long sarcoreg_i64;
#else
#include <math.h>
#include <stdint.h>
#define GLOBAL
#define INLINE static inline
typedef int64_t sarcoreg_i64;
#endif

#define SARCOREG_PI        3.14159265358979323846

#define SARCOREG_BILINEAR  0
#define SARCOREG_BICUBIC   1
#define SARCOREG_SINC      2
#define SARCOREG_MAX_TAPS  16

/* Node arrays packed in the nodes buffer, each ngl * ngs values. */
#define NODE_LAT_LO        0
#define NODE_LON_LO        1
#define NODE_U_LO          2
#define NODE_X_LO          3
#define NODE_LAT_HI        4
#define NODE_LON_HI        5
#define NODE_U_HI          6
#define NODE_X_HI          7
#define NODE_H             8
#define NODE_COUNT         9

/* Auxiliary outputs packed in the aux buffer, each rows * cols values. */
#define AUX_AZIMUTH_OFFSET 0
#define AUX_RANGE_OFFSET   1
#define AUX_HEIGHT         2
#define AUX_COUNT          3

INLINE int sarcoreg_iclamp(int v, int lo, int hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

INLINE double sarcoreg_sinc(double x)
{
    double px;

    if (fabs(x) < 1e-12)
        return 1.0;
    px = SARCOREG_PI * x;
    return sin(px) / px;
}

/* Interpolation weight of a sample at distance d from the target point. */
INLINE double sarcoreg_weight(int type, int taps, double d)
{
    double a = fabs(d);

    if (type == SARCOREG_BILINEAR)
        return a < 1.0 ? 1.0 - a : 0.0;
    if (type == SARCOREG_BICUBIC) {
        /* Keys cubic convolution, a = -0.5. */
        if (a <= 1.0)
            return 1.5 * a * a * a - 2.5 * a * a + 1.0;
        if (a < 2.0)
            return -0.5 * a * a * a + 2.5 * a * a - 4.0 * a + 2.0;
        return 0.0;
    }
    /* Sinc with a Hann window spanning the taps. */
    if (a >= 0.5 * taps)
        return 0.0;
    return sarcoreg_sinc(d) * (0.5 + 0.5 * cos(2.0 * SARCOREG_PI * d / taps));
}

/* Cell of v in the increasing node positions g[0..n-1] and the fraction
 * within it; values outside the nodes extrapolate from the edge cells. */
INLINE int sarcoreg_cell(GLOBAL const double *g, int n, double v, double *f)
{
    int lo = 0, hi = n - 1, mid;

    if (v <= g[0])
        lo = 0;
    else if (v >= g[n - 1])
        lo = n - 2;
    else {
        while (hi - lo > 1) {
            mid = (lo + hi) / 2;
            if (g[mid] <= v)
                lo = mid;
            else
                hi = mid;
        }
    }
    *f = (v - g[lo]) / (g[lo + 1] - g[lo]);
    return lo;
}

INLINE double sarcoreg_bilerp(GLOBAL const double *a, int ngs, int il, int is,
                              double fl, double fs)
{
    long k = (long)il * ngs + is;

    return (1.0 - fl) * ((1.0 - fs) * a[k] + fs * a[k + 1]) +
           fl * ((1.0 - fs) * a[k + ngs] + fs * a[k + ngs + 1]);
}

/* Bilinear DEM height at lat, lon; NaN outside the DEM or on no-data. */
INLINE double sarcoreg_dem(GLOBAL const float *dem, int nr, int nc, double lat0,
                           double lon0, double dlat, double dlon, double lat,
                           double lon)
{
    double fr = (lat - lat0) / dlat, fc = (lon - lon0) / dlon;
    int r0 = (int)floor(fr), c0 = (int)floor(fc);
    double v00, v01, v10, v11;
    long k;

    if (r0 < 0 || c0 < 0 || r0 + 1 >= nr || c0 + 1 >= nc)
        return NAN;
    fr -= r0;
    fc -= c0;
    k = (long)r0 * nc + c0;
    v00 = dem[k];
    v01 = dem[k + 1];
    v10 = dem[k + nc];
    v11 = dem[k + nc + 1];
    if (isnan(v00) || isnan(v01) || isnan(v10) || isnan(v11))
        return NAN;
    return (1.0 - fr) * ((1.0 - fc) * v00 + fc * v01) +
           fr * ((1.0 - fc) * v10 + fc * v11);
}

/* TOPS deramping and demodulation phase (SNAP Sentinel1Utils) of burst
 * line lb at range sample col; doppler holds kt, tref and fdc per sample. */
INLINE double sarcoreg_ramp(GLOBAL const double *doppler, int ncols, double dt,
                            double lb, int col)
{
    double ta = lb * dt;
    double kt = doppler[col];
    double dtr = ta - doppler[ncols + col];
    double fdc = doppler[2 * ncols + col];

    return -SARCOREG_PI * kt * dtr * dtr - 2.0 * SARCOREG_PI * fdc * ta;
}

/* Ramp phase at a fractional range position (parameters linear in range). */
INLINE double sarcoreg_ramp_frac(GLOBAL const double *doppler, int ncols,
                                 double dt, double lb, double x)
{
    int c0 = sarcoreg_iclamp((int)floor(x), 0, ncols - 1);
    int c1 = sarcoreg_iclamp(c0 + 1, 0, ncols - 1);
    double f = x - c0;
    double ta = lb * dt;
    double kt = (1.0 - f) * doppler[c0] + f * doppler[c1];
    double tref = (1.0 - f) * doppler[ncols + c0] + f * doppler[ncols + c1];
    double fdc =
        (1.0 - f) * doppler[2 * ncols + c0] + f * doppler[2 * ncols + c1];
    double dtr = ta - tref;

    return -SARCOREG_PI * kt * dtr * dtr - 2.0 * SARCOREG_PI * fdc * ta;
}

/* Coregister output pixel (r, c) of the reference segment.
 *
 * The reference pixel is burst line line0 + r, range sample c. Its
 * position in the secondary image, u (azimuth line index in the secondary
 * time frame) and x (range sample), is interpolated from the geometry
 * nodes; with a DEM the terrain height is solved first, the node values
 * being linear in height between h_lo and h_hi. The secondary burst holding
 * u is deramped, interpolated at (u, x) and rerampped there. */
INLINE void
sarcoreg_pixel(int r, int c, int cols, double line0, int ngl, int ngs,
               GLOBAL const double *grid_lines,
               GLOBAL const double *grid_samples, GLOBAL const double *nodes,
               int has_dem, double h_lo, double h_hi, GLOBAL const float *dem,
               int dem_rows, int dem_cols, double dem_lat0, double dem_lon0,
               double dem_dlat, double dem_dlon, int nseg, int sec_cols,
               GLOBAL const float *sec, GLOBAL const sarcoreg_i64 *segidx,
               GLOBAL const double *seginfo, GLOBAL const double *doppler,
               double dt, double shift, double range_shift, int ktype, int taps,
               GLOBAL float *out, GLOBAL float *aux, int want_aux, long npix)
{
    long idx = (long)r * cols + c;
    long n = (long)ngl * ngs;
    double line = line0 + r, fl, fs, u, x, h, lr;
    double wy[SARCOREG_MAX_TAPS], wx[SARCOREG_MAX_TAPS];
    double sr = 0.0, si = 0.0, sw = 0.0, ph, cs, sn;
    int il, is, j, s, i, q, bl, bc, row, col, nrow;
    long off;
    GLOBAL const float *data;
    GLOBAL const double *dop;

    out[2 * idx] = NAN;
    out[2 * idx + 1] = NAN;
    if (want_aux) {
        aux[AUX_AZIMUTH_OFFSET * npix + idx] = NAN;
        aux[AUX_RANGE_OFFSET * npix + idx] = NAN;
        aux[AUX_HEIGHT * npix + idx] = NAN;
    }

    il = sarcoreg_cell(grid_lines, ngl, line, &fl);
    is = sarcoreg_cell(grid_samples, ngs, (double)c, &fs);
    u = sarcoreg_bilerp(nodes + NODE_U_LO * n, ngs, il, is, fl, fs);
    x = sarcoreg_bilerp(nodes + NODE_X_LO * n, ngs, il, is, fl, fs);
    if (has_dem) {
        double lat0 =
            sarcoreg_bilerp(nodes + NODE_LAT_LO * n, ngs, il, is, fl, fs);
        double lon0 =
            sarcoreg_bilerp(nodes + NODE_LON_LO * n, ngs, il, is, fl, fs);
        double lat1 =
            sarcoreg_bilerp(nodes + NODE_LAT_HI * n, ngs, il, is, fl, fs);
        double lon1 =
            sarcoreg_bilerp(nodes + NODE_LON_HI * n, ngs, il, is, fl, fs);
        double w, hn;
        int it;

        h = sarcoreg_dem(dem, dem_rows, dem_cols, dem_lat0, dem_lon0, dem_dlat,
                         dem_dlon, lat0, lon0);
        if (isnan(h))
            return;
        /* The ground point moves with the height along the range circle:
         * iterate h = DEM(lat(h), lon(h)). */
        for (it = 0; it < 8; it++) {
            w = (h - h_lo) / (h_hi - h_lo);
            hn = sarcoreg_dem(dem, dem_rows, dem_cols, dem_lat0, dem_lon0,
                              dem_dlat, dem_dlon, lat0 + w * (lat1 - lat0),
                              lon0 + w * (lon1 - lon0));
            if (isnan(hn))
                return;
            if (fabs(hn - h) < 0.01) {
                h = hn;
                break;
            }
            h = hn;
        }
        w = (h - h_lo) / (h_hi - h_lo);
        u += w *
             (sarcoreg_bilerp(nodes + NODE_U_HI * n, ngs, il, is, fl, fs) - u);
        x += w *
             (sarcoreg_bilerp(nodes + NODE_X_HI * n, ngs, il, is, fl, fs) - x);
    }
    else
        h = sarcoreg_bilerp(nodes + NODE_H * n, ngs, il, is, fl, fs);
    u += shift;
    x += range_shift;
    if (want_aux) {
        aux[AUX_RANGE_OFFSET * npix + idx] = (float)(x - c);
        aux[AUX_HEIGHT * npix + idx] = (float)h;
    }

    /* Secondary segment (burst rows) holding the azimuth position. */
    j = -1;
    for (s = 0; s < nseg; s++) {
        if (u >= seginfo[2 * s] - 0.5 &&
            u < seginfo[2 * s] + segidx[2 * s + 1] - 0.5) {
            j = s;
            break;
        }
    }
    if (j < 0 || x < -0.5 || x > sec_cols - 0.5)
        return;
    off = segidx[2 * j];
    nrow = (int)segidx[2 * j + 1];
    data = sec + 2 * off;
    dop = doppler + 3L * j * sec_cols;
    lr = u - seginfo[2 * j];
    if (want_aux)
        aux[AUX_AZIMUTH_OFFSET * npix + idx] =
            (float)(seginfo[2 * j + 1] + lr - line);

    /* No output where the nearest secondary sample is null. */
    row = sarcoreg_iclamp((int)floor(lr + 0.5), 0, nrow - 1);
    col = sarcoreg_iclamp((int)floor(x + 0.5), 0, sec_cols - 1);
    if (isnan(data[2 * ((long)row * sec_cols + col)]))
        return;

    bl = (int)floor(lr) - taps / 2 + 1;
    bc = (int)floor(x) - taps / 2 + 1;
    for (i = 0; i < taps; i++) {
        wy[i] = sarcoreg_weight(ktype, taps, lr - (bl + i));
        wx[i] = sarcoreg_weight(ktype, taps, x - (bc + i));
    }
    for (i = 0; i < taps; i++) {
        GLOBAL const float *tdata = data;
        GLOBAL const double *tdop = dop;
        double lb;
        int ts;

        if (wy[i] == 0.0)
            continue;
        /* A tap beyond the segment reads the adjacent segment of the same
         * map (debursted image), deramped with its own burst; otherwise
         * the edge row of the segment is repeated. */
        row = bl + i;
        lb = seginfo[2 * j + 1] + row;
        if (row < 0 || row >= nrow) {
            long mrow = segidx[2 * j] / sec_cols + row;

            for (ts = 0; ts < nseg; ts++) {
                long r0 = segidx[2 * ts] / sec_cols;

                if (ts != j && mrow >= r0 && mrow < r0 + segidx[2 * ts + 1])
                    break;
            }
            if (ts < nseg) {
                row = (int)(mrow - segidx[2 * ts] / sec_cols);
                tdata = sec + 2 * segidx[2 * ts];
                tdop = doppler + 3L * ts * sec_cols;
                lb = seginfo[2 * ts + 1] + row;
            }
            else {
                row = sarcoreg_iclamp(row, 0, nrow - 1);
                lb = seginfo[2 * j + 1] + row;
            }
        }
        for (q = 0; q < taps; q++) {
            double w = wy[i] * wx[q], re, im;
            long k;

            if (w == 0.0)
                continue;
            col = sarcoreg_iclamp(bc + q, 0, sec_cols - 1);
            k = 2 * ((long)row * sec_cols + col);
            re = tdata[k];
            im = tdata[k + 1];
            if (isnan(re) || isnan(im))
                continue;
            ph = sarcoreg_ramp(tdop, sec_cols, dt, lb, col);
            cs = cos(ph);
            sn = sin(ph);
            sr += w * (re * cs - im * sn);
            si += w * (re * sn + im * cs);
            sw += w;
        }
    }
    if (sw <= 1e-6)
        return;
    sr /= sw;
    si /= sw;
    ph = sarcoreg_ramp_frac(dop, sec_cols, dt, seginfo[2 * j + 1] + lr, x);
    cs = cos(ph);
    sn = sin(ph);
    out[2 * idx] = (float)(sr * cs + si * sn);
    out[2 * idx + 1] = (float)(si * cs - sr * sn);
}

#endif /* SARCOREG_CORE_H */
