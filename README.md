# i.sar.coregistration

A [GRASS GIS](https://grass.osgeo.org/) addon that **coregisters a
Sentinel-1 TOPS SLC image onto a reference image**, both imported by
[r.in.s1slc](https://github.com/YannChemin/r.in.s1slc), following ESA
SNAP *Back-Geocoding* and *Enhanced Spectral Diversity*: geometric
coregistration from the orbits and a DEM, TOPS deramping around a
windowed-sinc interpolation, ESD refinement of the azimuth
coregistration in the burst overlaps, and optional range refinement by
cross-correlation. The per-pixel work runs on an OpenCL device (GPU or
CPU) or on the host CPU with OpenMP.

```sh
r.in.s1slc -b input=<reference>.zip output=s1_20230112 swath=IW2 polarization=VV bbox=...
r.in.s1slc -b input=<secondary>.zip output=s1_20230124 swath=IW2 polarization=VV bbox=...
i.sar.coregistration reference=s1_20230112_iw2_vv secondary=s1_20230124_iw2_vv \
    output=s1_20230124_co dem=dem.tif extra=elevation
```

Then *i.sar.interferometry* forms the interferogram and coherence.

## Features

| Feature | Option | Notes |
|---|---|---|
| Inputs | `reference=`, `secondary=` | burst maps (`r.in.s1slc -b`, recommended) or debursted maps |
| Geometry | `dem=`, `dem_height=geoid\|ellipsoid` | any GDAL DEM; EGM96 via PROJ grid; default annotation heights |
| Interpolation | `interpolation=sinc8` | sinc6/8/16 (Hann), bicubic, bilinear, on the deramped signal |
| ESD | default with burst maps, `-e` to disable | double-difference phase in burst overlaps, shift in metadata |
| Range refinement | `-r`, `xcorr_window=`, `xcorr_threshold=` | amplitude cross-correlation per burst, constant range shift |
| Extra maps | `extra=azimuth_offset,range_offset,elevation` | in reference geometry |
| Device | `device=auto\|gpu\|cpu\|host`, `platform=` | OpenCL (fp64 needed) or C/OpenMP, same code |

## Layout

| File | Role |
|---|---|
| `i.sar.coregistration.py` | GRASS module: metadata, orbit geometry, burst pairing, DEM, ESD, output |
| `sarcoreg_core.h` | per-pixel coregistration, compiled as C99 and as OpenCL C |
| `sarcoreg_kernels.cl` | OpenCL kernel entry point |
| `sarcoreg.c`, `sarcoreg.h` | `libsarcoreg`: device selection, host (OpenMP) and OpenCL execution, loaded through ctypes |
| `tests/s1sim.py` | simulator of Sentinel-1 TOPS SLC pairs written as SAFE products |

## Build and test

```sh
make MODULE_TOPDIR=$HOME/dev/grass
# The tests build libsarcoreg, simulate TOPS pairs and import them with
# r.in.s1slc (R_IN_S1SLC defaults to ~/dev/r.in.s1slc/r.in.s1slc.py):
grass --tmp-project XY --exec python3 -m pytest tests
```

The simulated pairs check the coherence and residual phase against the
simulated geometry, the ESD recovery of an injected orbit timing error,
the range refinement of an injected slant range timing error, the
DEM-assisted coregistration over a hill, debursted inputs, and that the
OpenCL and host paths agree.

## License

GPL-2.0-or-later.
