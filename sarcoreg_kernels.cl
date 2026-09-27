/* OpenCL entry point of i.sar.coregistration; the per-pixel code is
 * sarcoreg_core.h, prepended to this file when the kernels are embedded. */

__kernel void
coregister(int rows, int cols, double line0, int ngl, int ngs,
           __global const double *grid_lines,
           __global const double *grid_samples, __global const double *nodes,
           int has_dem, double h_lo, double h_hi, __global const float *dem,
           int dem_rows, int dem_cols, double dem_lat0, double dem_lon0,
           double dem_dlat, double dem_dlon, int nseg, int sec_cols,
           __global const float *sec, __global const long *segidx,
           __global const double *seginfo, __global const double *doppler,
           double dt, double shift, double range_shift, int ktype, int taps,
           __global float *out, __global float *aux, int want_aux)
{
    int c = get_global_id(0);
    int r = get_global_id(1);

    if (c >= cols || r >= rows)
        return;
    sarcoreg_pixel(r, c, cols, line0, ngl, ngs, grid_lines, grid_samples, nodes,
                   has_dem, h_lo, h_hi, dem, dem_rows, dem_cols, dem_lat0,
                   dem_lon0, dem_dlat, dem_dlon, nseg, sec_cols, sec, segidx,
                   seginfo, doppler, dt, shift, range_shift, ktype, taps, out,
                   aux, want_aux, (long)rows * cols);
}
