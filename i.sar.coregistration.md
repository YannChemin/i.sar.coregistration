## DESCRIPTION

*i.sar.coregistration* resamples a **secondary** Sentinel-1 TOPS (IW,
EW) Single Look Complex image onto the geometry of a **reference**
image of the same sub-swath and relative orbit, so that the two can be
combined pixel by pixel, e.g. by *i.sar.interferometry*. Both images
are complex maps imported by *r.in.s1slc* (`measure=complex`), with
their metadata.

The processing follows ESA SNAP (microwave toolbox) *Back-Geocoding*
and *Enhanced Spectral Diversity*:

- **Geometric coregistration**: every reference pixel is located on the
  ground from the reference orbit (zero-Doppler time, slant range and
  terrain height), and the ground point is projected into the secondary
  image with the secondary orbit. Orbit state vectors are interpolated
  with Lagrange polynomials over eight vectors. The terrain height comes
  from the **dem** or, without it, from the heights of the annotation
  geolocation grid. The positions are computed on a grid of nodes (every
  10 lines and 100 samples) for two heights enclosing the DEM range and
  interpolated to every pixel, where the height is solved from the DEM.
- **TOPS deramping**: the azimuth spectrum of a TOPS burst drifts along
  the burst (antenna steering). Before interpolation, the secondary
  signal is deramped and demodulated with the burst Doppler rate,
  reference time and Doppler centroid computed from the annotated
  azimuth FM rate, Doppler centroid polynomials and steering rate; after
  interpolation the ramp is restored at the interpolated position.
- **Interpolation**: a Hann-windowed sinc over 6, 8 or 16 samples, or
  bicubic or bilinear interpolation (**interpolation**). Taps on null
  samples are skipped and the weights renormalized; a pixel whose
  nearest secondary sample is null is null.
- **Enhanced spectral diversity** (ESD): with burst maps (see below),
  the residual azimuth misregistration is estimated in the overlaps of
  consecutive bursts, where each ground point is seen twice under looks
  whose Doppler frequencies differ by about kt × burst cycle time. The
  phase of the double difference interferogram
  (m<sub>k</sub> s<sub>k</sub><sup>\*</sup>)
  (m<sub>k+1</sub> s<sub>k+1</sub><sup>\*</sup>)<sup>\*</sup> is
  -2π Δf Δt<sub>az</sub> ε for a misregistration of ε lines. The phases
  are averaged with inverse quadratic weights γ²/(1-γ²) of the local
  phase consistency γ of the double difference (pixels with γ below
  **esd_coherence** are ignored), and the shift is added to the
  secondary positions before the final interpolation. Azimuth
  coregistration better than 0.01 line is needed to avoid phase jumps at
  the burst seams of TOPS interferograms, and orbits alone rarely give
  it.
- **Range refinement** (**-r** flag): residual range misregistration,
  e.g. from a slant range timing error, is estimated as in SNAP
  *Spectral Diversity*. Each burst is first coregistered in a window
  (**xcorr_window**, 512 × 512 by default, clipped to the valid area) at
  its centre; the de-meaned amplitudes of the reference and coregistered
  secondary windows are cross-correlated (normalized correlation over
  shifts of up to 8 samples, peak refined by FFT oversampling of the
  correlation by 128). The complex signals are oversampled by 2 in range
  before detection, which removes the bias of the amplitude correlation
  of speckle towards whole samples. Bursts whose correlation peak is
  below **xcorr_threshold** or whose shift exceeds one sample are
  discarded; the mean range shift of the others is added to all the
  secondary range positions, before ESD and the final interpolation. It
  also works on debursted images (one window per burst segment).

### Input images

The **reference** and **secondary** options are basenames of images
imported by *r.in.s1slc*, i.e. `{output}_{swath}_{pol}`:

- burst maps `{basename}_bNN_i` and `_q`, imported with the **-b** flag
  of *r.in.s1slc*: each burst is complete, including the lines it shares
  with its neighbours, which ESD needs. This is the recommended input.
  Reference and secondary bursts are paired by geometry (and their burst
  IDs are checked when both products have them); reference bursts that
  no secondary burst covers are skipped with a warning.
- debursted maps `{basename}_i` and `_q`: the coregistration uses the
  orbits only (no ESD). Around each burst seam, the rows between the
  reference and secondary seams hold data of different bursts in the two
  images, i.e. of different parts of the azimuth spectrum, and are poorly
  coherent.

Both images must be of the same kind. The module fails if a basename
matches both kinds of maps.

### Output

The coregistered secondary maps have the geometry of the reference maps
and their names, the reference basename being replaced by **output**:
`{output}_i` and `{output}_q`, or `{output}_bNN_i` and `{output}_bNN_q`
for each reference burst. They are FCELL maps (null where the secondary
image has no data) with the units, calibration, semantic label and
timestamp of the secondary image.

The **extra** option adds maps in reference geometry: the azimuth and
range offsets (secondary minus reference position, in lines of the
secondary burst and in samples) and the ellipsoidal terrain height used
(`elevation`), which *i.sar.interferometry* uses to remove the
topographic phase.

Each output map carries a `cell_misc/<map>/description.json` file with
the secondary product and annotation metadata (orbit, Doppler, bursts),
the reference raster geometry and a `coregistration` section: reference
and secondary maps and products, burst pairs, DEM, interpolation, the
ESD result (shift, and phase, Doppler separation, weight and shift of
each overlap) and the range refinement result (shift, and window,
azimuth and range offsets and correlation peak of each burst).

### DEM

The **dem** is any GDAL raster in any CRS covering the scene; it is
reprojected to a WGS84 longitude/latitude grid by bilinear
interpolation. Heights above the EGM96 geoid (**dem_height=geoid**, e.g.
SRTM, Copernicus DEM) are converted to ellipsoidal heights with the PROJ
grid `us_nga_egm96_15.tif` or `egm96_15.gtx`, which must be installed
(e.g. `projsync --file us_nga_egm96_15.tif`). Pixels where the DEM has
no data are null.

### Processing device

The per-pixel work (height solution, deramping, interpolation) runs in
the compute library `libsarcoreg`, on an OpenCL device or on the host
CPU with OpenMP (**device**), the same code being compiled for both.
OpenCL devices need double precision (`cl_khr_fp64`). The default,
**device=auto**, uses OpenCL whenever a device is usable: the OpenCL
GPUs first, then the OpenCL CPU devices; only without any, it falls
back to the host OpenMP code, with a warning giving the reason.
**platform** restricts OpenCL to the platforms whose name contains the
given text. **nprocs** sets the number of OpenMP threads of the host
path.

## NOTES

The current project must be unprojected (XY), like the *r.in.s1slc*
maps. The computational region is not used: every map is processed in
full.

The range refinement estimates one constant shift for the whole image.
It corrects timing errors, not the range offsets of the terrain relief:
without a DEM over hilly terrain it only absorbs their mean. Burst IDs
of products before IPF 3.4 are absent; pairing is then geometric only.

The module requires NumPy and the GDAL Python bindings, and a C compiler
with OpenMP and the OpenCL headers and ICD loader to build
`libsarcoreg`.

## EXAMPLES

Import two acquisitions burst by burst and coregister the second one on
the first, with the Copernicus DEM:

```sh
grass -c XY $HOME/grassdata/s1_sharjah -e
grass $HOME/grassdata/s1_sharjah/PERMANENT --exec bash

r.in.s1slc -b input=S1A_IW_SLC__1SDV_20230112T142507_20230112T142534_046752_059AD8_A55D.zip \
    output=s1_20230112 swath=IW2 polarization=VV bbox=55.35,25.25,55.50,25.40
r.in.s1slc -b input=S1A_IW_SLC__1SDV_20230124T...zip \
    output=s1_20230124 swath=IW2 polarization=VV bbox=55.35,25.25,55.50,25.40

i.sar.coregistration -r reference=s1_20230112_iw2_vv secondary=s1_20230124_iw2_vv \
    output=s1_20230124_co dem=copernicus_dem_30m.tif extra=elevation
```

The **-r** flag adds the range refinement by cross-correlation.

The ESD shift and the per-overlap estimates are in the metadata:

```sh
python3 -m json.tool $(g.gisenv get=GISDBASE)/$(g.gisenv get=LOCATION_NAME)/PERMANENT/cell_misc/s1_20230124_co_b04_i/description.json | grep -A3 '"esd"'
```

## REFERENCES

- N. Yagüe-Martínez, P. Prats-Iraola, F. Rodríguez González, R. Brcic,
  R. Shau, D. Geudtner, M. Eineder and R. Bamler, *Interferometric
  Processing of Sentinel-1 TOPS Data*, IEEE Transactions on Geoscience
  and Remote Sensing, 54(4), 2220-2234, 2016,
  doi:10.1109/TGRS.2015.2497902.
- ESA, *Sentinel-1 Product Specification*, S1-RS-MDA-52-7441.
- ESA SNAP microwave toolbox:
  <https://github.com/senbox-org/microwave-toolbox>

## SEE ALSO

*[i.sar.interferometry](i.sar.interferometry.html),
[r.in.s1slc](r.in.s1slc.html)*

## AUTHORS

Yann Chemin
