#!/usr/bin/env python3

# MODULE:    i.sar.coregistration
# AUTHOR(S): Yann Chemin
# PURPOSE:   Coregisters a secondary Sentinel-1 TOPS SLC image onto a
#            reference image, both imported by r.in.s1slc, from the orbits
#            and an optional DEM (SNAP Back-Geocoding), with TOPS deramping
#            and enhanced spectral diversity (ESD) azimuth refinement.
# COPYRIGHT: (C) 2026 by Yann Chemin and the GRASS Development Team
# SPDX-License-Identifier: GPL-2.0-or-later

# %module
# % description: Coregisters a Sentinel-1 TOPS SLC image onto a reference SLC image (geometric coregistration and ESD).
# % keyword: imagery
# % keyword: SAR
# % keyword: radar
# % keyword: Sentinel-1
# % keyword: SLC
# % keyword: TOPS
# % keyword: coregistration
# % keyword: interferometry
# % keyword: OpenCL
# %end

# %option
# % key: reference
# % type: string
# % required: yes
# % key_desc: basename
# % label: Basename of the reference image imported by r.in.s1slc
# % description: E.g. s1_20230112_iw2_vv for maps s1_20230112_iw2_vv_i/_q, or their burst maps s1_20230112_iw2_vv_bNN_i/_q (-b import)
# %end

# %option
# % key: secondary
# % type: string
# % required: yes
# % key_desc: basename
# % label: Basename of the secondary image imported by r.in.s1slc
# %end

# %option G_OPT_R_BASENAME_OUTPUT
# % required: yes
# % description: Basename of the coregistered secondary maps (_i, _q, with the burst tags of the reference)
# %end

# %option G_OPT_F_INPUT
# % key: dem
# % required: no
# % label: Digital elevation model file (any GDAL raster, any CRS)
# % description: Default: heights of the annotation geolocation grid
# % guisection: DEM
# %end

# %option
# % key: dem_height
# % type: string
# % required: no
# % options: geoid,ellipsoid
# % answer: geoid
# % label: Height reference of the DEM
# % descriptions: geoid;Heights above the EGM96 geoid (e.g. SRTM, Copernicus DEM), converted to ellipsoidal heights;ellipsoid;Heights above the WGS84 ellipsoid
# % guisection: DEM
# %end

# %option
# % key: interpolation
# % type: string
# % required: no
# % options: bilinear,bicubic,sinc6,sinc8,sinc16
# % answer: sinc8
# % label: Interpolation of the deramped secondary signal
# % description: sincN is a Hann-windowed sinc over N samples
# % guisection: Processing
# %end

# %option
# % key: extra
# % type: string
# % required: no
# % multiple: yes
# % options: azimuth_offset,range_offset,elevation
# % label: Additional maps in reference geometry
# % descriptions: azimuth_offset;Secondary minus reference azimuth position (lines);range_offset;Secondary minus reference range position (samples);elevation;Ellipsoidal terrain height used (meters)
# % guisection: Output
# %end

# %option
# % key: esd_coherence
# % type: double
# % required: no
# % answer: 0.3
# % options: 0-1
# % label: Coherence threshold of the pixels used by ESD
# % guisection: Processing
# %end

# %option
# % key: device
# % type: string
# % required: no
# % options: auto,gpu,cpu,host
# % answer: auto
# % label: Processing device
# % descriptions: auto;First OpenCL GPU, else OpenCL CPU, else host;gpu;OpenCL GPU;cpu;OpenCL CPU device;host;C code on the host CPU (OpenMP)
# % guisection: Processing
# %end

# %option
# % key: platform
# % type: string
# % required: no
# % label: OpenCL platform name filter (case-insensitive substring)
# % guisection: Processing
# %end

# %option G_OPT_M_NPROCS
# % guisection: Processing
# %end

# %option
# % key: xcorr_window
# % type: integer
# % required: no
# % multiple: yes
# % key_desc: range,azimuth
# % answer: 512,512
# % label: Cross-correlation window for the range refinement (samples, lines)
# % description: One window at the centre of each burst, clipped to its valid area
# % guisection: Processing
# %end

# %option
# % key: xcorr_threshold
# % type: double
# % required: no
# % answer: 0.1
# % options: 0-1
# % label: Minimum correlation peak of a burst used by the range refinement
# % guisection: Processing
# %end

# %flag
# % key: e
# % label: Do not refine the azimuth coregistration by ESD
# %end

# %flag
# % key: r
# % label: Refine the range coregistration by cross-correlation
# % description: Estimates a constant range shift from the amplitudes of the coregistered bursts (SNAP Spectral Diversity range shift)
# %end

import atexit
import ctypes
import json
import os
import sys
from datetime import datetime, timedelta

import grass.script as gs

C = 299792458.0
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_B = WGS84_A * (1.0 - WGS84_F)
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

# Largest residual range shift (samples) accepted from a burst by the
# cross-correlation (SNAP maxRangeShift).
MAX_RANGE_SHIFT = 1.0
# Half size of the correlation chip oversampled around the peak, and the
# oversampling factor (SNAP fine registration accuracy and oversampling).
XCORR_CHIP = 8
XCORR_OVERSAMPLING = 128

# Spacing of the geometry nodes (lines, samples); the node values are
# interpolated bilinearly, the mapping being smooth.
NODE_LINES = 10
NODE_SAMPLES = 100
# Height margin (meters) around the DEM range for the geometry nodes.
DEM_MARGIN = 50.0
# Coherence estimation window of ESD (lines, samples).
ESD_WINDOW = (3, 10)

KERNELS = {
    "bilinear": (0, 2),
    "bicubic": (1, 4),
    "sinc6": (2, 6),
    "sinc8": (2, 8),
    "sinc16": (2, 16),
}
EXTRA_UNITS = {
    "azimuth_offset": "lines",
    "range_offset": "samples",
    "elevation": "meters",
}
AUX_INDEX = {"azimuth_offset": 0, "range_offset": 1, "elevation": 2}
N_NODE_ARRAYS = 9

TMP_FILES = []


def cleanup():
    for path in TMP_FILES:
        try:
            os.remove(path)
        except OSError:
            pass


def parse_time(text):
    """Parse an ISO UTC time with any number of fractional digits."""
    text = text.strip().rstrip("Z")
    if "." in text:
        head, frac = text.split(".")
        frac = (frac + "000000")[:6]
    else:
        head, frac = text, "000000"
    return datetime.strptime(head, "%Y-%m-%dT%H:%M:%S").replace(microsecond=int(frac))


class TimeAxis:
    """Seconds since a common epoch, exact to the microsecond."""

    def __init__(self, epoch):
        self.epoch = epoch

    def sec(self, text):
        return (parse_time(text) - self.epoch) // timedelta(microseconds=1) * 1e-6


# Geodesy.


def geodetic_to_ecef(lat, lon, h):
    import numpy as np

    lat = np.radians(lat)
    lon = np.radians(lon)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    return np.stack(
        [
            (n + h) * np.cos(lat) * np.cos(lon),
            (n + h) * np.cos(lat) * np.sin(lon),
            (n * (1.0 - WGS84_E2) + h) * np.sin(lat),
        ],
        axis=-1,
    )


def ecef_to_geodetic(p):
    """Return latitude, longitude (degrees) and ellipsoidal height."""
    import numpy as np

    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    lon = np.arctan2(y, x)
    rho = np.hypot(x, y)
    lat = np.arctan2(z, rho * (1.0 - WGS84_E2))
    for _i in range(6):
        n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
        h = rho / np.cos(lat) - n
        lat = np.arctan2(z, rho * (1.0 - WGS84_E2 * n / (n + h)))
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    h = rho / np.cos(lat) - n
    return np.degrees(lat), np.degrees(lon), h


class Orbit:
    """Orbit state vectors with Lagrange interpolation (SNAP OrbitStateVectors)."""

    def __init__(self, times, pos, vel, npoints=8):
        import numpy as np

        order = np.argsort(times)
        self.t = np.asarray(times, dtype=np.float64)[order]
        self.pos = np.asarray(pos, dtype=np.float64)[order]
        self.vel = np.asarray(vel, dtype=np.float64)[order]
        if self.t.size < 2:
            gs.fatal(_("At least two orbit state vectors are needed"))
        self.m = min(npoints, self.t.size)

    def _weights(self, t):
        import numpy as np

        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        n, m = self.t.size, self.m
        i0 = np.clip(np.searchsorted(self.t, t) - m // 2, 0, n - m)
        idx = i0[:, None] + np.arange(m)
        tt = self.t[idx]
        num = np.broadcast_to((t[:, None] - tt)[:, None, :], (t.size, m, m)).copy()
        den = tt[:, :, None] - tt[:, None, :]
        eye = np.eye(m, dtype=bool)
        num[:, eye] = 1.0
        den[:, eye] = 1.0
        return idx, np.prod(num / den, axis=2)

    def state(self, t):
        import numpy as np

        idx, w = self._weights(t)
        pos = np.einsum("ij,ijk->ik", w, self.pos[idx])
        vel = np.einsum("ij,ijk->ik", w, self.vel[idx])
        return pos, vel

    def acceleration(self, t, h=0.01):
        return (self.state(t + h)[1] - self.state(t - h)[1]) / (2.0 * h)


def zero_doppler(orbit, points, t0):
    """Zero-Doppler time and slant range of ECEF points (Newton iterations)."""
    import numpy as np

    t = np.array(t0, dtype=np.float64)
    for _i in range(30):
        s, v = orbit.state(t)
        a = orbit.acceleration(t)
        d = points - s
        f = np.einsum("ij,ij->i", d, v)
        fp = -np.einsum("ij,ij->i", v, v) + np.einsum("ij,ij->i", d, a)
        step = -f / fp
        t = t + step
        if np.max(np.abs(step)) < 1e-11:
            break
    s, _v = orbit.state(t)
    return t, np.linalg.norm(points - s, axis=1)


def range_doppler_to_ecef(orbit, t, rng, h, guess):
    """Ground point at zero-Doppler time t, slant range rng and ellipsoidal
    height h (DORIS lph2xyz): Newton on the Doppler, range and ellipsoid
    equations, the ellipsoid axes being corrected until the geodetic height
    of the point is h."""
    import numpy as np

    s, v = orbit.state(t)
    p = np.array(guess, dtype=np.float64)
    h = np.broadcast_to(np.asarray(h, dtype=np.float64), t.shape)
    dh = np.zeros_like(h)
    for _outer in range(3):
        ah = WGS84_A + h + dh
        bh = WGS84_B + h + dh
        for _i in range(12):
            d = p - s
            f = np.stack(
                [
                    np.einsum("ij,ij->i", d, v),
                    np.einsum("ij,ij->i", d, d) - rng**2,
                    (p[:, 0] ** 2 + p[:, 1] ** 2) / ah**2 + p[:, 2] ** 2 / bh**2 - 1.0,
                ],
                axis=1,
            )
            jac = np.stack(
                [
                    v,
                    2.0 * d,
                    np.stack(
                        [2 * p[:, 0] / ah**2, 2 * p[:, 1] / ah**2, 2 * p[:, 2] / bh**2],
                        1,
                    ),
                ],
                axis=1,
            )
            step = np.linalg.solve(jac, -f[:, :, None])[:, :, 0]
            p += step
            if np.max(np.abs(step)) < 1e-5:
                break
        dh += h - ecef_to_geodetic(p)[2]
    return p


# Input images.


def read_json(path):
    with open(path) as fd:
        return json.load(fd)


def map_meta(name):
    found = gs.find_file(name, element="cell")
    if not found["file"]:
        gs.fatal(_("Raster map <{}> not found").format(name))
    env = gs.gisenv()
    path = os.path.join(
        env["GISDBASE"],
        env["LOCATION_NAME"],
        found["mapset"],
        "cell_misc",
        found["name"],
        "description.json",
    )
    if not os.path.isfile(path):
        gs.fatal(
            _("Raster map <{}> has no r.in.s1slc metadata ({})").format(name, path)
        )
    meta = read_json(path)
    for key in ("product", "swath", "raster_geometry"):
        if key not in meta:
            gs.fatal(
                _(
                    "Metadata of <{}> lack the '{}' section; not an r.in.s1slc map"
                ).format(name, key)
            )
    if "segments" not in meta["raster_geometry"]:
        gs.fatal(
            _(
                "Metadata of <{}> have no row segments; re-import the product "
                "with a current r.in.s1slc"
            ).format(name)
        )
    return meta


def exists(name):
    return bool(gs.find_file(name, element="cell")["file"])


class Segment:
    """Rows of a map read from consecutive lines of one burst."""

    def __init__(self, image, map_index, seg):
        self.image = image
        self.map_index = map_index
        self.burst = seg["burst"] - 1
        self.burst_id = seg["burst_id"]
        self.first_row = seg["first_row"]
        self.rows = seg["rows"]
        self.first_line = seg["first_line"]


class SlcImage:
    """A complex image imported by r.in.s1slc: debursted or burst maps."""

    def __init__(self, base, role):
        self.base = base
        self.role = role
        debursted = exists(base + "_i") and exists(base + "_q")
        burst_maps = sorted(
            m.split("@")[0]
            for m in gs.list_strings("raster", pattern=base + "_b[0-9][0-9]_i")
        )
        if debursted and burst_maps:
            gs.fatal(
                _(
                    "Both debursted maps <{b}_i> and burst maps <{b}_bNN_i> exist; "
                    "remove one set or import them with different output names"
                ).format(b=base)
            )
        if not debursted and not burst_maps:
            gs.fatal(
                _(
                    "No complex maps <{b}_i>, <{b}_q> or <{b}_bNN_i>, <{b}_bNN_q> found ({r} image)"
                ).format(b=base, r=role)
            )
        stems = [base] if debursted else [m[: -len("_i")] for m in burst_maps]
        self.debursted = debursted
        self.maps = []
        for stem in stems:
            if not exists(stem + "_q"):
                gs.fatal(_("Raster map <{}> not found").format(stem + "_q"))
            self.maps.append(
                {
                    "stem": stem,
                    "i": stem + "_i",
                    "q": stem + "_q",
                    "meta": map_meta(stem + "_i"),
                }
            )
        first = self.maps[0]["meta"]
        self.product = first["product"]
        self.swath = first["swath"]
        self.meta = first
        for m in self.maps[1:]:
            if m["meta"]["product"].get("product_name") != self.product.get(
                "product_name"
            ):
                gs.fatal(
                    _("Burst maps of <{}> come from different products").format(base)
                )
        self.segments = []
        for k, m in enumerate(self.maps):
            geom = m["meta"]["raster_geometry"]
            m["rows"] = geom["rows"]
            m["cols"] = geom["cols"]
            m["segments"] = [Segment(self, k, s) for s in geom["segments"]]
            self.segments += m["segments"]
        self._data = {}

    def setup(self, axis):
        """Time-dependent parameters, in seconds of the common time axis."""
        import numpy as np

        sw = self.swath
        self._axis = axis
        self.dt = float(sw["azimuth_time_interval"])
        self.lpb = int(sw["lines_per_burst"])
        self.ns = int(sw["number_of_samples"])
        self.fs = float(sw["range_sampling_rate"])
        self.srt = float(sw["slant_range_time"])
        self.wavelength = float(sw["wavelength"])
        self.burst_t = np.array([axis.sec(b["azimuth_time"]) for b in sw["bursts"]])
        self.t_origin = float(self.burst_t[0])
        osv = sw["orbit_state_vectors"]
        if not osv:
            gs.fatal(
                _("No orbit state vectors in the metadata of <{}>").format(self.base)
            )
        self.orbit = Orbit(
            [axis.sec(o["time"]) for o in osv],
            [o["position"] for o in osv],
            [o["velocity"] for o in osv],
        )
        self.first_line_time = axis.sec(sw["product_first_line_utc_time"])
        self.last_line_time = axis.sec(sw["product_last_line_utc_time"])
        grid = sw["geolocation_grid"]
        times = sorted({axis.sec(p["azimuth_time"]) for p in grid})
        pixels = sorted({int(p["pixel"]) for p in grid})
        self.grid_t = np.array(times)
        self.grid_p = np.array(pixels, dtype=np.float64)
        shape = (len(times), len(pixels))
        self.grid_lat = np.full(shape, np.nan)
        self.grid_lon = np.full(shape, np.nan)
        self.grid_h = np.full(shape, np.nan)
        ti = {t: i for i, t in enumerate(times)}
        pi = {p: j for j, p in enumerate(pixels)}
        for p in grid:
            i, j = ti[axis.sec(p["azimuth_time"])], pi[int(p["pixel"])]
            self.grid_lat[i, j] = p["latitude"]
            self.grid_lon[i, j] = p["longitude"]
            self.grid_h[i, j] = p["height"]
        if np.isnan(self.grid_lat).any() or len(times) < 2 or len(pixels) < 2:
            gs.fatal(
                _("Incomplete geolocation grid in the metadata of <{}>").format(
                    self.base
                )
            )

    def grid_interp(self, t, pixel):
        """Bilinear latitude, longitude and height of the annotation grid."""
        import numpy as np

        def axis_weights(axis_values, v):
            i = np.clip(np.searchsorted(axis_values, v) - 1, 0, axis_values.size - 2)
            f = (v - axis_values[i]) / (axis_values[i + 1] - axis_values[i])
            return i, f

        i, fi = axis_weights(self.grid_t, t)
        j, fj = axis_weights(self.grid_p, pixel)
        out = []
        for g in (self.grid_lat, self.grid_lon, self.grid_h):
            out.append(
                (1 - fi) * ((1 - fj) * g[i, j] + fj * g[i, j + 1])
                + fi * ((1 - fj) * g[i + 1, j] + fj * g[i + 1, j + 1])
            )
        return out

    def line_time(self, k, line):
        return self.burst_t[k] + line * self.dt

    def slant_range(self, sample):
        return 0.5 * C * (self.srt + sample / self.fs)

    def ground(self, k, line, sample, height=None):
        """ECEF points of burst k lines and samples (arrays of equal size),
        on the annotation heights unless height is given."""
        import numpy as np

        t = self.line_time(k, line)
        lat, lon, hgrid = self.grid_interp(t, sample)
        h = hgrid if height is None else np.broadcast_to(height, t.shape)
        guess = geodetic_to_ecef(lat, lon, h)
        return range_doppler_to_ecef(
            self.orbit, t, self.slant_range(sample), h, guess
        ), hgrid

    def doppler(self, k):
        """TOPS deramping parameters of burst k per range sample: Doppler
        rate kt, reference time tref and Doppler centroid fdc (SNAP
        Sentinel1Utils computeDopplerRate/computeReferenceTime)."""
        import numpy as np

        sw = self.swath
        cache = self._data.setdefault("doppler", {})
        if k in cache:
            return cache[k]
        axis_t = self._axis
        tau = self.srt + np.arange(self.ns) / self.fs
        t_mid = self.line_time(k, 0.5 * self.lpb)

        def nearest(entries):
            if not entries:
                gs.fatal(
                    _("Missing Doppler annotation in the metadata of <{}>").format(
                        self.base
                    )
                )
            return min(
                entries, key=lambda e: abs(axis_t.sec(e["azimuth_time"]) - t_mid)
            )

        fm = nearest(sw["azimuth_fm_rate"])
        dt0 = tau - fm["t0"]
        c = fm["polynomial"]
        ka = c[0] + c[1] * dt0 + c[2] * dt0**2
        dc = nearest(sw["doppler_centroid"])
        method = sw.get("processing", {}).get("dcMethod", "")
        poly = (
            dc["data_dc_polynomial"]
            if "Data Analysis" in method
            else dc["geometry_dc_polynomial"]
        )
        dt0 = tau - dc["t0"]
        fdc = sum(coef * dt0**i for i, coef in enumerate(poly))
        _s, vel = self.orbit.state(
            np.array([0.5 * (self.first_line_time + self.last_line_time)])
        )
        v = float(np.linalg.norm(vel[0]))
        krot = (
            2.0 * v * np.radians(float(sw["azimuth_steering_rate"])) / self.wavelength
        )
        kt = ka * krot / (ka - krot)
        fvs = [s for b in sw["bursts"] for s in b["first_valid_sample"] if s >= 0]
        fvp = min(max(fvs) if fvs else 0, self.ns - 1)
        tref = self.lpb * self.dt / 2.0 + fdc[fvp] / ka[fvp] - fdc / ka
        cache[k] = (kt, tref, fdc)
        return cache[k]

    def read(self, map_index):
        """Complex data of one map as interleaved float32 (rows, cols, 2)."""
        import numpy as np

        if map_index in self._data:
            return self._data[map_index]
        m = self.maps[map_index]
        out = np.empty((m["rows"], m["cols"], 2), dtype=np.float32)
        for c, key in enumerate(("i", "q")):
            out[:, :, c] = read_raster(m[key], m["rows"], m["cols"])
        self._data[map_index] = out
        return out

    def release(self, map_index):
        self._data.pop(map_index, None)


def read_raster(name, rows, cols):
    import numpy as np

    path = gs.tempfile(create=False)
    TMP_FILES.append(path)
    env = os.environ.copy()
    env["GRASS_REGION"] = gs.region_env(raster=name)
    info = gs.region(env=env)
    if (int(info["rows"]), int(info["cols"])) != (rows, cols):
        gs.fatal(
            _("Raster map <{}> is {}x{}, its metadata say {}x{}").format(
                name, info["rows"], info["cols"], rows, cols
            )
        )
    gs.run_command(
        "r.out.bin",
        flags="f",
        input=name,
        output=path,
        bytes=4,
        null="nan",
        quiet=True,
        env=env,
    )
    data = np.fromfile(path, dtype=np.float32).reshape(rows, cols)
    os.remove(path)
    TMP_FILES.remove(path)
    return data


def write_raster(name, data, title):
    """Write a float array (rows, cols) with NaN nulls as an FCELL map."""
    import numpy as np

    rows, cols = data.shape
    path = gs.tempfile(create=False)
    TMP_FILES.append(path)
    np.ascontiguousarray(data, dtype=np.float32).tofile(path)
    gs.run_command(
        "r.in.bin",
        flags="f",
        input=path,
        output=name,
        title=title,
        bytes=4,
        order="native",
        north=rows,
        south=0,
        east=cols,
        west=0,
        rows=rows,
        cols=cols,
        anull="nan",
        overwrite=gs.overwrite(),
        quiet=True,
    )
    os.remove(path)
    TMP_FILES.remove(path)


# Compute library.


class SarcoregJob(ctypes.Structure):
    """Mirror of struct sarcoreg_job (sarcoreg.h)."""

    _fields_ = [
        ("rows", ctypes.c_int),
        ("cols", ctypes.c_int),
        ("line0", ctypes.c_double),
        ("ngl", ctypes.c_int),
        ("ngs", ctypes.c_int),
        ("grid_lines", ctypes.POINTER(ctypes.c_double)),
        ("grid_samples", ctypes.POINTER(ctypes.c_double)),
        ("nodes", ctypes.POINTER(ctypes.c_double)),
        ("has_dem", ctypes.c_int),
        ("h_lo", ctypes.c_double),
        ("h_hi", ctypes.c_double),
        ("dem", ctypes.POINTER(ctypes.c_float)),
        ("dem_rows", ctypes.c_int),
        ("dem_cols", ctypes.c_int),
        ("dem_lat0", ctypes.c_double),
        ("dem_lon0", ctypes.c_double),
        ("dem_dlat", ctypes.c_double),
        ("dem_dlon", ctypes.c_double),
        ("nseg", ctypes.c_int),
        ("sec_cols", ctypes.c_int),
        ("sec", ctypes.POINTER(ctypes.c_float)),
        ("segidx", ctypes.POINTER(ctypes.c_int64)),
        ("seginfo", ctypes.POINTER(ctypes.c_double)),
        ("doppler", ctypes.POINTER(ctypes.c_double)),
        ("dt", ctypes.c_double),
        ("shift", ctypes.c_double),
        ("range_shift", ctypes.c_double),
        ("kernel_type", ctypes.c_int),
        ("taps", ctypes.c_int),
        ("out", ctypes.POINTER(ctypes.c_float)),
        ("aux", ctypes.POINTER(ctypes.c_float)),
        ("want_aux", ctypes.c_int),
    ]


def find_library():
    name = "libsarcoreg.so"
    candidates = []
    if os.environ.get("I_SAR_COREGISTRATION_LIB"):
        candidates.append(os.environ["I_SAR_COREGISTRATION_LIB"])
    candidates.append(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), name))
    for base in (os.environ.get("GRASS_ADDON_BASE"), os.environ.get("GISBASE")):
        if base:
            candidates.append(os.path.join(base, "etc", "i.sar.coregistration", name))
    for path in candidates:
        if os.path.isfile(path):
            return path
    gs.fatal(
        _("Compute library {} not found in: {}").format(name, ", ".join(candidates))
    )


class Engine:
    def __init__(self, device, platform, nprocs):
        path = find_library()
        self.lib = ctypes.CDLL(path)
        self.lib.sarcoreg_init.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
        ]
        self.lib.sarcoreg_run.argtypes = [
            ctypes.POINTER(SarcoregJob),
            ctypes.c_char_p,
            ctypes.c_int,
        ]
        self.msg = ctypes.create_string_buffer(8192)
        if self.lib.sarcoreg_init(
            device.encode(), (platform or "").encode(), nprocs, self.msg, 8192
        ):
            gs.fatal(
                _("Processing device <{}> unavailable: {}").format(
                    device, self.msg.value.decode()
                )
            )
        self.device = self.msg.value.decode()
        if device == "auto" and "no OpenCL" in self.device:
            gs.warning(
                _("No usable OpenCL device, falling back to OpenMP: {}").format(
                    self.device
                )
            )

    def run(self, job):
        if self.lib.sarcoreg_run(ctypes.byref(job), self.msg, 8192):
            gs.fatal(_("Coregistration failed: {}").format(self.msg.value.decode()))

    def finish(self):
        self.lib.sarcoreg_finish()


def ptr(array, ctype):
    return array.ctypes.data_as(ctypes.POINTER(ctype))


# DEM.


class Dem:
    """DEM resampled to a WGS84 longitude/latitude grid over the scene,
    in ellipsoidal heights."""

    def __init__(self, path, height_ref, bbox):
        import numpy as np
        from osgeo import gdal

        gdal.UseExceptions()
        west, south, east, north = bbox
        try:
            src = gdal.Open(path)
        except RuntimeError as e:
            gs.fatal(_("Unable to open DEM <{}>: {}").format(path, e))
        ds = gdal.Warp(
            "",
            src,
            format="MEM",
            dstSRS="EPSG:4326",
            outputBounds=(west, south, east, north),
            resampleAlg="bilinear",
            dstNodata=float("nan"),
            outputType=gdal.GDT_Float32,
        )
        data = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
        gt = ds.GetGeoTransform()
        if np.all(np.isnan(data)):
            gs.fatal(_("DEM <{}> does not cover the scene ({})").format(path, bbox))
        if height_ref == "geoid":
            geoid = self._geoid(gdal, ds)
            data = data + geoid
        self.data = np.ascontiguousarray(data)
        self.rows, self.cols = data.shape
        self.lat0 = gt[3] + 0.5 * gt[5]
        self.lon0 = gt[0] + 0.5 * gt[1]
        self.dlat = gt[5]
        self.dlon = gt[1]
        self.h_lo = float(np.nanmin(data)) - DEM_MARGIN
        self.h_hi = float(np.nanmax(data)) + DEM_MARGIN
        self.path = path
        self.height_ref = height_ref
        missing = float(np.mean(np.isnan(data)))
        if missing > 0:
            gs.warning(
                _(
                    "DEM <{}> has no data over {:.1f}% of the scene box; those pixels are null"
                ).format(path, 100 * missing)
            )

    @staticmethod
    def _geoid(gdal, like):
        from osgeo import osr

        names = ("us_nga_egm96_15.tif", "egm96_15.gtx")
        dirs = list(osr.GetPROJSearchPaths() or []) + [
            "/usr/share/proj",
            "/usr/local/share/proj",
        ]
        for d in dirs:
            for n in names:
                path = os.path.join(d, n)
                if os.path.isfile(path):
                    gt = like.GetGeoTransform()
                    ds = gdal.Warp(
                        "",
                        path,
                        format="MEM",
                        dstSRS="EPSG:4326",
                        outputBounds=(
                            gt[0],
                            gt[3] + like.RasterYSize * gt[5],
                            gt[0] + like.RasterXSize * gt[1],
                            gt[3],
                        ),
                        width=like.RasterXSize,
                        height=like.RasterYSize,
                        resampleAlg="bilinear",
                        outputType=gdal.GDT_Float32,
                    )
                    return ds.GetRasterBand(1).ReadAsArray()
        gs.fatal(
            _(
                "EGM96 geoid grid ({}) not found in the PROJ data directories {}; "
                "install it (e.g. projsync --file us_nga_egm96_15.tif) or use "
                "dem_height=ellipsoid with an ellipsoidal DEM"
            ).format(" or ".join(names), dirs)
        )


# Coregistration.


class Coregistration:
    def __init__(self, ref, sec, dem, engine, kernel, want_aux):
        self.ref = ref
        self.sec = sec
        self.dem = dem
        self.engine = engine
        self.kernel = kernel
        self.want_aux = want_aux
        self._nodes = {}
        # Constant range shift (samples) added to the secondary positions.
        self.range_shift = 0.0

    def nodes(self, seg):
        """Geometry nodes of the reference burst of a segment: positions of
        the reference burst lines and samples in the secondary image."""
        import numpy as np

        key = (seg.burst, seg.first_line, seg.rows)
        if key in self._nodes:
            return self._nodes[key]
        ref, sec = self.ref, self.sec
        l0, l1 = seg.first_line, seg.first_line + seg.rows - 1
        lines = np.unique(np.r_[np.arange(l0, l1, NODE_LINES), l1]).astype(np.float64)
        samples = np.unique(
            np.r_[np.arange(0, ref.ns - 1, NODE_SAMPLES), ref.ns - 1]
        ).astype(np.float64)
        if lines.size < 2:
            lines = np.array([l0 - 0.5, l0 + 0.5])
        ll, ss = np.meshgrid(lines, samples, indexing="ij")
        ll, ss = ll.ravel(), ss.ravel()
        t = ref.line_time(seg.burst, ll)
        heights = [None] if self.dem is None else [self.dem.h_lo, self.dem.h_hi]
        arrays = np.zeros((N_NODE_ARRAYS, ll.size))
        t_guess = t - ref.first_line_time + sec.first_line_time
        for h_index, height in enumerate(heights):
            p, hgrid = ref.ground(seg.burst, ll, ss, height)
            lat, lon, h = ecef_to_geodetic(p)
            ts, rs = zero_doppler(sec.orbit, p, t_guess)
            u = (ts - sec.t_origin) / sec.dt
            x = (2.0 * rs / C - sec.srt) * sec.fs
            base = 4 * h_index
            arrays[base : base + 4] = lat, lon, u, x
            if h_index == 0:
                arrays[8] = h
        if self.dem is None:
            arrays[4:8] = arrays[0:4]
        result = (lines, samples, np.ascontiguousarray(arrays))
        self._nodes[key] = result
        return result

    def sec_centre_time(self, seg):
        """Secondary zero-Doppler time of the centre of a reference segment."""
        import numpy as np

        ref = self.ref
        line = np.array([seg.first_line + 0.5 * (seg.rows - 1)])
        p, _h = ref.ground(seg.burst, line, np.array([0.5 * (ref.ns - 1)]))
        t = ref.line_time(seg.burst, line)
        ts, _r = zero_doppler(
            self.sec.orbit, p, t - ref.first_line_time + self.sec.first_line_time
        )
        return float(ts[0])

    def match_burst(self, seg):
        """Secondary burst map covering a reference burst map, or None."""
        sec = self.sec
        ts = self.sec_centre_time(seg)
        best = None
        for s in sec.segments:
            t0 = sec.line_time(s.burst, s.first_line)
            t1 = sec.line_time(s.burst, s.first_line + s.rows - 1)
            if t0 <= ts <= t1:
                d = abs(ts - 0.5 * (t0 + t1))
                if best is None or d < best[0]:
                    best = (d, s)
        if best is None:
            return None
        s = best[1]
        if (
            seg.burst_id is not None
            and s.burst_id is not None
            and seg.burst_id != s.burst_id
        ):
            gs.fatal(
                _(
                    "Reference burst {} (burst ID {}) matches secondary burst {} by "
                    "geometry but its burst ID is {}; inconsistent products"
                ).format(seg.burst + 1, seg.burst_id, s.burst + 1, s.burst_id)
            )
        return s

    def secondary_input(self, sec_segments):
        """Secondary data and per-segment arrays for the compute library."""
        import numpy as np

        sec = self.sec
        maps = sorted({s.map_index for s in sec_segments})
        if len(maps) != 1:
            gs.fatal(_("Internal error: secondary segments span several maps"))
        data = sec.read(maps[0])
        cols = data.shape[1]
        segidx = np.array(
            [[s.first_row * cols, s.rows] for s in sec_segments], dtype=np.int64
        )
        seginfo = np.array(
            [
                [
                    (sec.line_time(s.burst, s.first_line) - sec.t_origin) / sec.dt,
                    s.first_line,
                ]
                for s in sec_segments
            ],
            dtype=np.float64,
        )
        doppler = np.ascontiguousarray(
            np.stack([np.stack(sec.doppler(s.burst)) for s in sec_segments]),
            dtype=np.float64,
        )
        return data, segidx, seginfo, doppler

    def run(self, seg, sec_segments, row0, rows, shift):
        """Coregister reference burst lines seg.first_line + row0 ... + rows - 1."""
        import numpy as np

        lines, samples, nodes = self.nodes(seg)
        data, segidx, seginfo, doppler = self.secondary_input(sec_segments)
        cols = self.ref.ns
        out = np.empty((rows, cols, 2), dtype=np.float32)
        aux = (
            np.empty((3, rows, cols), dtype=np.float32)
            if self.want_aux
            else np.empty(1, dtype=np.float32)
        )
        dem = self.dem
        dem_data = dem.data if dem is not None else np.zeros(1, dtype=np.float32)
        job = SarcoregJob(
            rows=rows,
            cols=cols,
            line0=float(seg.first_line + row0),
            ngl=lines.size,
            ngs=samples.size,
            grid_lines=ptr(lines, ctypes.c_double),
            grid_samples=ptr(samples, ctypes.c_double),
            nodes=ptr(nodes, ctypes.c_double),
            has_dem=int(dem is not None),
            h_lo=dem.h_lo if dem is not None else 0.0,
            h_hi=dem.h_hi if dem is not None else 1.0,
            dem=ptr(dem_data, ctypes.c_float),
            dem_rows=dem.rows if dem is not None else 1,
            dem_cols=dem.cols if dem is not None else 1,
            dem_lat0=dem.lat0 if dem is not None else 0.0,
            dem_lon0=dem.lon0 if dem is not None else 0.0,
            dem_dlat=dem.dlat if dem is not None else 1.0,
            dem_dlon=dem.dlon if dem is not None else 1.0,
            nseg=len(sec_segments),
            sec_cols=data.shape[1],
            sec=ptr(data, ctypes.c_float),
            segidx=ptr(segidx, ctypes.c_int64),
            seginfo=ptr(seginfo, ctypes.c_double),
            doppler=ptr(doppler, ctypes.c_double),
            dt=self.sec.dt,
            shift=shift,
            range_shift=self.range_shift,
            kernel_type=self.kernel[0],
            taps=self.kernel[1],
            out=ptr(out, ctypes.c_float),
            aux=ptr(aux, ctypes.c_float),
            want_aux=int(self.want_aux),
        )
        self.engine.run(job)
        return out[..., 0] + 1j * out[..., 1], (aux if self.want_aux else None)


def boxcar(a, wy, wx):
    """Moving average over a wy x wx window (edges shrink the window)."""
    import numpy as np

    def along(x, w, axis):
        x = np.moveaxis(x, axis, 0)
        c = np.cumsum(
            np.concatenate([np.zeros((1,) + x.shape[1:], x.dtype), x]), axis=0
        )
        n = x.shape[0]
        lo = np.clip(np.arange(n) - w // 2, 0, n)
        hi = np.clip(np.arange(n) - w // 2 + w, 0, n)
        shape = (-1,) + (1,) * (x.ndim - 1)
        out = (c[hi] - c[lo]) / (hi - lo).reshape(shape)
        return np.moveaxis(out, 0, axis)

    return along(along(a, wy, 0), wx, 1)


def consistency(z, window):
    """Local |<z>| / <|z|> of a complex array (1 for a constant phase)."""
    import numpy as np

    valid = np.isfinite(z)
    z = np.where(valid, z, 0)
    num = np.abs(boxcar(z, *window))
    den = boxcar(np.abs(z), *window)
    with np.errstate(invalid="ignore", divide="ignore"):
        gamma = np.where(den > 0, num / den, 0.0)
    return np.where(valid, gamma, 0.0)


def cross_correlate(m, s, search):
    """Offset (lines, samples) of s relative to m, s(x + offset) ~ m(x),
    and the correlation peak, by normalized cross-correlation of the
    de-meaned amplitudes (SNAP/DORIS crossCorrelateFFT): integer shifts up
    to search (in range, of the signal oversampled by 2), then a chip
    around the peak oversampled by FFT."""
    import numpy as np

    # Oversample the complex signals by 2 in range before detection: the
    # amplitude of critically sampled speckle is aliased, which biases the
    # correlation peak towards whole samples. Only range is oversampled,
    # its spectrum being baseband (the TOPS azimuth spectrum drifts).
    valid = np.isfinite(m) & np.isfinite(s)
    am = np.abs(oversample_range(np.where(valid, m, 0)))
    asec = np.abs(oversample_range(np.where(valid, s, 0)))
    valid = np.repeat(valid, 2, axis=1)
    valid[:, 1:-1] &= valid[:, :-2] & valid[:, 2:]
    if valid.sum() < 16:
        return None
    am = np.where(valid, am - am[valid].mean(), 0.0)
    asec = np.where(valid, asec - asec[valid].mean(), 0.0)
    ones = valid.astype(np.float64)
    rows, cols = am.shape
    shape = (2 * rows, 2 * cols)

    def correlate(a, b):
        """sum over x of a(x) b(x + t), for every shift t (index t mod shape)."""
        return np.fft.ifft2(np.conj(np.fft.fft2(a, shape)) * np.fft.fft2(b, shape)).real

    cross = correlate(am, asec)
    norm_m = correlate(am**2, ones)
    norm_s = correlate(ones, asec**2)
    with np.errstate(invalid="ignore", divide="ignore"):
        ncc = np.where(
            (norm_m > 0) & (norm_s > 0), cross / np.sqrt(norm_m * norm_s), 0.0
        )
    ncc = np.fft.fftshift(ncc)
    cl, cp = rows, cols  # zero shift after fftshift
    sl, sp = min(search, rows // 4), min(search, cols // 4)
    window = ncc[cl - sl : cl + sl + 1, cp - sp : cp + sp + 1]
    il, ip = np.unravel_index(np.argmax(window), window.shape)
    pl, pp = cl - sl + il, cp - sp + ip
    # Oversample a chip around the integer peak.
    h = XCORR_CHIP
    chip = ncc[pl - h : pl + h, pp - h : pp + h]
    n = 2 * h
    spec = np.fft.fftshift(np.fft.fft2(chip))
    big = n * XCORR_OVERSAMPLING
    pad = np.zeros((big, big), dtype=complex)
    o = (big - n) // 2
    pad[o : o + n, o : o + n] = spec
    fine = np.fft.ifft2(np.fft.ifftshift(pad)).real * XCORR_OVERSAMPLING**2
    fl, fp = np.unravel_index(np.argmax(fine), fine.shape)
    peak = float(fine[fl, fp])
    return (
        pl - h + fl / XCORR_OVERSAMPLING - cl,
        0.5 * (pp - h + fp / XCORR_OVERSAMPLING - cp),
        peak,
    )


def oversample_range(z):
    """Complex signal oversampled by 2 along range by FFT zero padding."""
    import numpy as np

    n = z.shape[1]
    spec = np.fft.fft(z, axis=1)
    out = np.zeros((z.shape[0], 2 * n), dtype=complex)
    half = n // 2
    out[:, :half] = spec[:, :half]
    out[:, -(n - half) :] = spec[:, half:]
    return np.fft.ifft(out, axis=1) * 2


def range_refinement(co, plan, window, threshold):
    """Constant range shift (samples) to add to the secondary positions,
    averaged over the bursts (SNAP Spectral Diversity range shift): each
    burst is coregistered in a window at its centre and cross-correlated
    with the reference."""
    import numpy as np

    ref = co.ref
    search = XCORR_CHIP
    bursts = []
    for seg, secs in plan:
        bm = ref.swath["bursts"][seg.burst]
        first = max(seg.first_line, bm["first_valid_line"])
        last = min(seg.first_line + seg.rows - 1, bm["last_valid_line"])
        fvs = [v for v in bm["first_valid_sample"] if v >= 0]
        lvs = [v for v in bm["last_valid_sample"] if v >= 0]
        if last < first or not fvs:
            continue
        c0, c1 = max(fvs), min(lvs)
        rows = min(window[1], last - first + 1)
        cols = min(window[0], c1 - c0 + 1)
        if rows < 4 * XCORR_CHIP or cols < 4 * XCORR_CHIP:
            bursts.append(
                {"reference_burst": seg.burst + 1, "reason": "window too small"}
            )
            continue
        r0 = (first + last + 1 - rows) // 2
        x0 = (c0 + c1 + 1 - cols) // 2
        m = complex_rows(ref, seg, r0 - seg.first_line, rows)[:, x0 : x0 + cols]
        s, _x = co.run(seg, secs, r0 - seg.first_line, rows, 0.0)
        result = cross_correlate(m, s[:, x0 : x0 + cols], search)
        entry = {"reference_burst": seg.burst + 1, "window": [cols, rows]}
        if result is None:
            entry["reason"] = "no valid pixels"
        else:
            dl, dp, peak = result
            entry.update({"azimuth_offset": dl, "range_offset": dp, "peak": peak})
            entry["used"] = bool(peak >= threshold and abs(dp) <= MAX_RANGE_SHIFT)
        bursts.append(entry)
    used = [b["range_offset"] for b in bursts if b.get("used")]
    return (float(np.mean(used)) if used else None), bursts


def esd(co, pairs, threshold):
    """Azimuth shift (lines) to add to the secondary positions, estimated by
    enhanced spectral diversity over the burst overlaps (SNAP Spectral
    Diversity, average estimator with inverse quadratic coherence weights,
    the coherence being the local phase consistency of the double
    difference).

    For consecutive reference bursts k and k+1 the double-difference
    interferogram (m_k s_k*)(m_k+1 s_k+1*)* in their overlap has the phase
    -2 pi eps dt df, df = kt (t_k+1 - t_k) being the Doppler separation of
    the two looks and eps the azimuth misregistration (lines)."""
    import numpy as np

    ref = co.ref
    overlaps = []
    for (sa, ma), (sb, mb) in pairs:
        ka, kb = sa.burst, sb.burst
        n_off = int(round((ref.burst_t[kb] - ref.burst_t[ka]) / ref.dt))
        bursts = ref.swath["bursts"]
        first = max(
            bursts[ka]["first_valid_line"], n_off + bursts[kb]["first_valid_line"]
        )
        last = min(bursts[ka]["last_valid_line"], n_off + bursts[kb]["last_valid_line"])
        if last - first + 1 < ESD_WINDOW[0]:
            continue
        rows = last - first + 1
        m_a = complex_rows(ref, sa, first, rows)
        m_b = complex_rows(ref, sb, first - n_off, rows)
        s_a, _x = co.run(sa, [ma], first - sa.first_line, rows, 0.0)
        s_b, _x = co.run(sb, [mb], first - n_off - sb.first_line, rows, 0.0)
        i_a = m_a * np.conj(s_a)
        i_b = m_b * np.conj(s_b)
        e = i_a * np.conj(i_b)
        # The double difference cancels the flat-earth and topographic
        # fringes: its local phase consistency measures the coherence.
        gamma = np.minimum(consistency(e, ESD_WINDOW), 0.999)
        w = np.where(
            (gamma >= threshold) & np.isfinite(e) & (np.abs(e) > 0),
            gamma**2 / (1 - gamma**2),
            0.0,
        )
        total = float(w.sum())
        entry = {
            "reference_bursts": [ka + 1, kb + 1],
            "lines": rows,
            "weight": total,
            "coherent_pixels": int(np.count_nonzero(w)),
        }
        if total > 0:
            kt = ref.doppler(ka)[0]
            df = kt * (ref.burst_t[kb] - ref.burst_t[ka])
            unit = np.where(w > 0, e / np.where(np.abs(e) > 0, np.abs(e), 1), 0)
            phase = float(np.angle(np.sum(w * unit)))
            df_mean = float(np.sum(w * df[None, :]) / total)
            entry["phase"] = phase
            entry["doppler_separation"] = df_mean
            entry["shift"] = phase / (2 * np.pi * df_mean * ref.dt)
            entry["search_boundary"] = 0.5 / abs(df_mean * ref.dt)
        overlaps.append(entry)
    good = [o for o in overlaps if "shift" in o]
    if not good:
        return None, overlaps
    total = sum(o["weight"] for o in good)
    shift = sum(o["shift"] * o["weight"] for o in good) / total
    return shift, overlaps


def complex_rows(image, seg, row0, rows):
    data = image.read(seg.map_index)
    block = data[seg.first_row + row0 : seg.first_row + row0 + rows]
    return block[..., 0] + 1j * block[..., 1]


def check_pair(ref, sec):
    rs, ss = ref.swath, sec.swath
    for key, label in (
        ("swath", "sub-swath"),
        ("polarization", "polarization"),
        ("mode", "mode"),
    ):
        if rs.get(key) != ss.get(key):
            gs.fatal(
                _("Reference and secondary differ in {}: {} and {}").format(
                    label, rs.get(key), ss.get(key)
                )
            )
    ro_r = ref.product.get("relative_orbit_start")
    ro_s = sec.product.get("relative_orbit_start")
    if ro_r and ro_s and ro_r != ro_s:
        gs.fatal(
            _(
                "Reference and secondary are on different relative orbits ({} and {})"
            ).format(ro_r, ro_s)
        )
    if abs(float(rs["radar_frequency"]) - float(ss["radar_frequency"])) > 1.0:
        gs.fatal(_("Reference and secondary have different radar frequencies"))


def scene_bbox(ref, margin=0.02):
    import numpy as np

    return (
        float(np.min(ref.grid_lon)) - margin,
        float(np.min(ref.grid_lat)) - margin,
        float(np.max(ref.grid_lon)) + margin,
        float(np.max(ref.grid_lat)) + margin,
    )


def read_timestamp(name):
    try:
        return gs.read_command("r.timestamp", map=name, quiet=True).strip()
    except gs.CalledModuleError:
        return ""


def main():
    import numpy as np

    options, flags = gs.parser()
    atexit.register(cleanup)

    if int(gs.region()["projection"]) != 0:
        gs.fatal(
            _(
                "SLC images are in radar geometry: the current project must be unprojected (XY)"
            )
        )

    ref = SlcImage(options["reference"], "reference")
    sec = SlcImage(options["secondary"], "secondary")
    check_pair(ref, sec)
    if ref.debursted != sec.debursted:
        gs.fatal(
            _(
                "Reference and secondary must both be debursted or both burst maps "
                "(r.in.s1slc -b); <{}> is {}, <{}> is {}"
            ).format(
                ref.base,
                "debursted" if ref.debursted else "burst maps",
                sec.base,
                "debursted" if sec.debursted else "burst maps",
            )
        )
    epoch = parse_time(ref.swath["product_first_line_utc_time"]).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    axis = TimeAxis(epoch)
    for image in (ref, sec):
        image.setup(axis)

    extra = options["extra"].split(",") if options["extra"] else []
    use_esd = not flags["e"]
    if use_esd and ref.debursted:
        gs.warning(
            _(
                "ESD needs whole bursts with their overlaps (import with r.in.s1slc -b); "
                "the azimuth coregistration of these debursted images relies on the "
                "orbits only and phase jumps may appear at the burst seams"
            )
        )
        use_esd = False

    prefix = options["output"]
    if ref.debursted:
        jobs = [(ref.maps[0], prefix)]
    else:
        jobs = [(m, prefix + m["stem"][len(ref.base) :]) for m in ref.maps]
    planned = []
    for _m, stem in jobs:
        planned += [stem + "_i", stem + "_q"] + [stem + "_" + e for e in extra]
    if not gs.overwrite():
        for name in planned:
            if gs.find_file(name, element="cell", mapset=".")["file"]:
                gs.fatal(
                    _(
                        "Raster map <{}> already exists, use --overwrite to replace it"
                    ).format(name)
                )

    dem = None
    if options["dem"]:
        gs.message(_("Preparing the DEM..."))
        dem = Dem(options["dem"], options["dem_height"], scene_bbox(ref))

    nprocs = int(options["nprocs"] or 0)
    if nprocs < 0:
        nprocs = max(1, (os.cpu_count() or 1) + nprocs)
    engine = Engine(options["device"], options["platform"], nprocs)
    gs.message(_("Processing on {}").format(engine.device))
    co = Coregistration(
        ref, sec, dem, engine, KERNELS[options["interpolation"]], bool(extra)
    )

    # Pair each reference segment with the secondary segments it may use.
    plan = []
    if ref.debursted:
        for seg in ref.segments:
            plan.append((seg, sec.segments))
    else:
        for seg in ref.segments:
            match = co.match_burst(seg)
            if match is None:
                gs.warning(
                    _(
                        "Reference burst {}: no secondary burst covers it, skipped"
                    ).format(seg.burst + 1)
                )
                continue
            plan.append((seg, [match]))
    if not plan:
        gs.fatal(_("No reference burst is covered by the secondary image"))

    range_meta = {"applied": False}
    if flags["r"]:
        window = [int(v) for v in options["xcorr_window"].split(",")]
        if len(window) != 2 or min(window) < 4 * XCORR_CHIP:
            gs.fatal(
                _(
                    "xcorr_window must be two integers of at least {} (range,azimuth)"
                ).format(4 * XCORR_CHIP)
            )
        gs.message(_("Estimating the range shift by cross-correlation..."))
        threshold = float(options["xcorr_threshold"])
        estimate, bursts = range_refinement(co, plan, window, threshold)
        range_meta.update({"bursts": bursts, "window": window, "threshold": threshold})
        if estimate is None:
            gs.warning(
                _(
                    "Range cross-correlation failed in every burst (peak below {} or "
                    "shift beyond {} sample); range shift not applied"
                ).format(threshold, MAX_RANGE_SHIFT)
            )
        else:
            co.range_shift = estimate
            range_meta.update({"applied": True, "shift": estimate})
            gs.message(_("Range shift: {:.4f} samples").format(estimate))

    shift = 0.0
    esd_meta = {"applied": False}
    if use_esd:
        pairs = []
        by_burst = {seg.burst: (seg, secs[0]) for seg, secs in plan}
        for k in sorted(by_burst):
            if (
                k + 1 in by_burst
                and by_burst[k + 1][1].burst == by_burst[k][1].burst + 1
            ):
                pairs.append((by_burst[k], by_burst[k + 1]))
        if not pairs:
            gs.warning(
                _(
                    "ESD needs at least two consecutive bursts; azimuth coregistration from the orbits only"
                )
            )
            esd_meta["reason"] = "no consecutive bursts"
        else:
            gs.message(
                _(
                    "Estimating the azimuth shift by ESD over {} burst overlaps..."
                ).format(len(pairs))
            )
            estimate, overlaps = esd(co, pairs, float(options["esd_coherence"]))
            esd_meta.update(
                {
                    "overlaps": overlaps,
                    "coherence_threshold": float(options["esd_coherence"]),
                }
            )
            if estimate is None:
                gs.warning(
                    _(
                        "ESD found no coherent pixels in the burst overlaps; shift not applied"
                    )
                )
                esd_meta["reason"] = "no coherent pixels"
            else:
                shift = estimate
                esd_meta.update({"applied": True, "shift": shift})
                gs.message(_("ESD azimuth shift: {:.5f} lines").format(shift))
                bound = min(
                    o["search_boundary"] for o in overlaps if "search_boundary" in o
                )
                if abs(shift) > 0.5 * bound:
                    gs.warning(
                        _(
                            "ESD shift {:.4f} lines is close to the ambiguity limit ({:.4f}); "
                            "check the orbits"
                        ).format(shift, bound)
                    )

    # Coregister and write the maps of each reference map.
    kernel_name = options["interpolation"]
    for ref_map, stem in jobs:
        rows, cols = ref_map["rows"], ref_map["cols"]
        result = np.full((rows, cols), np.nan + 1j * np.nan, dtype=np.complex64)
        aux = np.full((3, rows, cols), np.nan, dtype=np.float32) if extra else None
        pairs_meta = []
        for seg, secs in plan:
            if seg.map_index != ref_map["segments"][0].map_index:
                continue
            gs.message(
                _("Coregistering reference burst {} ({} lines)...").format(
                    seg.burst + 1, seg.rows
                )
            )
            if ref.debursted:
                nodes_u = co.nodes(seg)[2][[2, 6]]
                # Keep the adjacent segments read by the interpolation taps.
                margin = 2 + co.kernel[1]
                lo, hi = (
                    float(np.min(nodes_u)) - margin,
                    float(np.max(nodes_u)) + margin,
                )
                secs = [
                    s
                    for s in secs
                    if (sec.line_time(s.burst, s.first_line) - sec.t_origin) / sec.dt
                    <= hi
                    and (
                        sec.line_time(s.burst, s.first_line + s.rows - 1) - sec.t_origin
                    )
                    / sec.dt
                    >= lo
                ]
                if not secs:
                    continue
            values, extra_values = co.run(seg, secs, 0, seg.rows, shift)
            result[seg.first_row : seg.first_row + seg.rows] = values
            if extra:
                aux[:, seg.first_row : seg.first_row + seg.rows] = extra_values
            pairs_meta.append(
                {
                    "reference_burst": seg.burst + 1,
                    "secondary_bursts": sorted({s.burst + 1 for s in secs}),
                }
            )
        write_outputs(
            options,
            ref,
            sec,
            ref_map,
            stem,
            result,
            aux,
            extra,
            dem,
            kernel_name,
            esd_meta,
            range_meta,
            pairs_meta,
        )
        done = ref_map["segments"][0].map_index
        ref.release(done)
        # Keep in memory only the secondary maps still needed.
        needed = {
            s.map_index for seg, secs in plan if seg.map_index > done for s in secs
        }
        for index in range(len(sec.maps)):
            if index not in needed:
                sec.release(index)
    engine.finish()
    return 0


def write_outputs(
    options,
    ref,
    sec,
    ref_map,
    stem,
    result,
    aux,
    extra,
    dem,
    kernel_name,
    esd_meta,
    range_meta,
    pairs_meta,
):
    import numpy as np

    env = gs.gisenv()
    mapset_dir = os.path.join(env["GISDBASE"], env["LOCATION_NAME"], env["MAPSET"])
    sec_meta = sec.meta
    sw = sec.swath
    mission = sec.product.get("mission") or sw.get("mission")
    stamp = read_timestamp(sec.maps[0]["i"])
    coreg_meta = {
        "reference": ref.base,
        "reference_maps": [ref_map["i"], ref_map["q"]],
        "reference_product": ref.product.get("product_name"),
        "secondary": sec.base,
        "secondary_maps": [m["i"] for m in sec.maps] + [m["q"] for m in sec.maps],
        "secondary_product": sec.product.get("product_name"),
        "method": "geometric (orbits, zero-Doppler)"
        + (", DEM" if dem else ", annotation terrain height"),
        "dem": dem.path if dem else None,
        "dem_height": dem.height_ref if dem else None,
        "interpolation": kernel_name,
        "tops_deramping": True,
        "burst_pairs": pairs_meta,
        "esd": esd_meta,
        "range_refinement": range_meta,
    }
    outputs = [
        ("i", result.real, "calibrated amplitude"),
        ("q", result.imag, "calibrated amplitude"),
    ]
    if sec_meta.get("calibration", "none") == "none":
        outputs = [("i", result.real, "DN"), ("q", result.imag, "DN")]
    for e in extra:
        outputs.append((e, aux[AUX_INDEX[e]], EXTRA_UNITS[e]))
    valid = np.isfinite(result.real)
    gs.message(
        _("Coregistered <{}>: {:.1f}% valid pixels").format(
            stem, 100.0 * float(np.mean(valid))
        )
    )
    for key, data, units in outputs:
        name = "{}_{}".format(stem, key)
        if key in ("i", "q"):
            title = "{} {} SLC {} {} {} coregistered on {}".format(
                mission,
                sw.get("mode"),
                sw.get("swath"),
                sw.get("polarization"),
                key,
                ref.product.get("product_name"),
            )
            label = "S1_{}_{}".format(sw.get("polarization"), key.upper())
        else:
            title = "{} {} coregistration {} ({} on {})".format(
                mission,
                sw.get("swath"),
                key.replace("_", " "),
                sec.product.get("product_name"),
                ref.product.get("product_name"),
            )
            label = "S1_{}".format(key)
        write_raster(name, data, title)
        gs.run_command(
            "r.support",
            map=name,
            title=title,
            units=units,
            source1=sec.product.get("product_name") or "",
            source2="reference: {}".format(ref.product.get("product_name")),
            description="Coregistered by i.sar.coregistration ({}); metadata in cell_misc/{}/description.json".format(
                "; ".join(
                    [
                        "ESD shift {:.5f} lines".format(esd_meta["shift"])
                        if esd_meta.get("applied")
                        else "no ESD",
                        "range shift {:.4f} samples".format(range_meta["shift"])
                        if range_meta.get("applied")
                        else "no range refinement",
                    ]
                ),
                name,
            ),
            semantic_label=label,
            quiet=True,
        )
        if stamp:
            gs.run_command("r.timestamp", map=name, date=stamp, quiet=True)
        gs.raster_history(name, overwrite=True)
        meta = {
            "product": sec.product,
            "swath": sw,
            "raster_geometry": ref_map["meta"]["raster_geometry"],
            "coregistration": coreg_meta,
        }
        if key in ("i", "q"):
            meta.update(
                {
                    "measure": key,
                    "calibration": sec_meta.get("calibration"),
                    "absolute_calibration_constant": sec_meta.get(
                        "absolute_calibration_constant"
                    ),
                    "thermal_noise_removed": sec_meta.get("thermal_noise_removed"),
                }
            )
        else:
            meta["measure"] = key
        meta_dir = os.path.join(mapset_dir, "cell_misc", name)
        os.makedirs(meta_dir, exist_ok=True)
        with open(os.path.join(meta_dir, "description.json"), "w") as fd:
            json.dump(meta, fd, indent=1)


if __name__ == "__main__":
    sys.exit(main())
