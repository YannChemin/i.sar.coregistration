"""Tests of i.sar.coregistration on simulated Sentinel-1 TOPS pairs."""

import numpy as np
import pytest

import grass.script as gs

from conftest import (
    coherence,
    description,
    opencl_available,
    read_complex,
    read_map,
    run_module,
    truth_phase,
)
import s1sim

N_OFF = int(round(s1sim.CYCLE_LINES))


def burst_coherences(session, pair, ref, stem):
    out = []
    for k in range(s1sim.NBURSTS):
        m = read_complex(session, f"{ref}_b{k + 1:02d}")
        s = read_complex(session, f"{stem}_b{k + 1:02d}")
        phase, _h = truth_phase(pair, k, 0, s1sim.LPB)
        out.append(coherence(m, s, phase))
    return out


def seam_phases(session, ref, stem):
    """Phase difference of consecutive burst interferograms in their overlap."""
    out = []
    for k in (1, 2):
        a = read_complex(session, f"{ref}_b{k:02d}") * np.conj(
            read_complex(session, f"{stem}_b{k:02d}")
        )
        b = read_complex(session, f"{ref}_b{k + 1:02d}") * np.conj(
            read_complex(session, f"{stem}_b{k + 1:02d}")
        )
        lines = slice(N_OFF + 3, s1sim.LPB - 2)
        e = a[lines] * np.conj(b[3 : s1sim.LPB - 2 - N_OFF])
        out.append(float(np.angle(np.nansum(e))))
    return out


def test_esd_recovers_orbit_timing_error(project, pairs, library):
    pair, _dem = pairs["flat"]
    proc = run_module(
        project,
        library,
        reference="flat_ref_iw1_vv",
        secondary="flat_sec_iw1_vv",
        output="flat_co",
    )
    assert proc.returncode == 0, proc.stderr
    esd = description(project, "flat_co_b01_i")["coregistration"]["esd"]
    assert esd["applied"]
    # The secondary orbit times are late by 0.05 line: its positions are
    # 0.05 line too far and ESD must bring them back.
    assert esd["shift"] == pytest.approx(-0.05, abs=0.005)
    assert len(esd["overlaps"]) == 2
    for coh, phase in burst_coherences(project, pair, "flat_ref_iw1_vv", "flat_co"):
        assert coh > 0.98
        assert abs(phase) < 0.05
    assert all(
        abs(p) < 0.04 for p in seam_phases(project, "flat_ref_iw1_vv", "flat_co")
    )


def test_without_esd_seams_jump(project, pairs, library):
    proc = run_module(
        project,
        library,
        "e",
        reference="flat_ref_iw1_vv",
        secondary="flat_sec_iw1_vv",
        output="flat_noesd",
    )
    assert proc.returncode == 0, proc.stderr
    assert (
        description(project, "flat_noesd_b02_i")["coregistration"]["esd"]["applied"]
        is False
    )
    # 0.05 line misregistration: 2 pi 0.05 dt kt T_cycle, about 0.12 rad.
    assert all(
        abs(p) > 0.08 for p in seam_phases(project, "flat_ref_iw1_vv", "flat_noesd")
    )


def test_range_refinement(project, pairs, library):
    """The secondary slant range time is annotated 0.3 sample late: its
    positions are 0.3 sample short and the cross-correlation must add it."""
    pair, _dem = pairs["range"]
    common = {
        "reference": "range_ref_iw1_vv",
        "secondary": "range_sec_iw1_vv",
    }
    proc = run_module(project, library, "r", output="range_co", **common)
    assert proc.returncode == 0, proc.stderr
    proc = run_module(project, library, output="range_raw", **common)
    assert proc.returncode == 0, proc.stderr
    meta = description(project, "range_co_b02_i")["coregistration"]
    refinement = meta["range_refinement"]
    assert refinement["applied"]
    assert refinement["shift"] == pytest.approx(0.3, abs=0.05)
    assert len(refinement["bursts"]) == 3
    assert all(b["used"] and b["peak"] > 0.5 for b in refinement["bursts"])
    assert description(project, "range_raw_b02_i")["coregistration"][
        "range_refinement"
    ] == {"applied": False}
    refined = burst_coherences(project, pair, "range_ref_iw1_vv", "range_co")
    raw = burst_coherences(project, pair, "range_ref_iw1_vv", "range_raw")
    for (c1, _p1), (c0, _p0) in zip(refined, raw):
        assert c1 > 0.985
        assert c1 > c0 + 0.03


def test_range_refinement_without_error(project, library):
    proc = run_module(
        project,
        library,
        "r",
        reference="flat_ref_iw1_vv",
        secondary="flat_sec_iw1_vv",
        output="flat_r",
    )
    assert proc.returncode == 0, proc.stderr
    refinement = description(project, "flat_r_b01_i")["coregistration"][
        "range_refinement"
    ]
    assert abs(refinement["shift"]) < 0.02


def test_opencl_matches_host(project, library):
    if not opencl_available(library):
        pytest.skip("no OpenCL device with double precision")
    for device, output in (("host", "cmp_host"), ("auto", "cmp_ocl")):
        proc = run_module(
            project,
            library,
            reference="flat_ref_iw1_vv",
            secondary="flat_sec_iw1_vv",
            output=output,
            device=device,
            extra="azimuth_offset,range_offset,elevation",
        )
        assert proc.returncode == 0, proc.stderr
    for suffix in (
        "b02_i",
        "b02_q",
        "b02_azimuth_offset",
        "b02_range_offset",
        "b02_elevation",
    ):
        a = read_map(project, "cmp_host_" + suffix)
        b = read_map(project, "cmp_ocl_" + suffix)
        np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
        np.testing.assert_allclose(
            a[np.isfinite(a)], b[np.isfinite(b)], rtol=1e-4, atol=1e-3
        )


def test_dem(project, pairs, library):
    pair, dem = pairs["hill"]
    common = {
        "reference": "hill_ref_iw1_vv",
        "secondary": "hill_sec_iw1_vv",
    }
    proc = run_module(
        project,
        library,
        output="hill_co",
        dem=dem,
        dem_height="ellipsoid",
        extra="elevation",
        **common,
    )
    assert proc.returncode == 0, proc.stderr
    proc = run_module(project, library, output="hill_nodem", **common)
    assert proc.returncode == 0, proc.stderr
    with_dem = burst_coherences(project, pair, "hill_ref_iw1_vv", "hill_co")
    without = burst_coherences(project, pair, "hill_ref_iw1_vv", "hill_nodem")
    for (c1, p1), (c0, _p0) in zip(with_dem, without):
        # 1 km baseline: about 0.9 on these slopes when the terrain is
        # known, much less when the range offsets ignore it.
        assert c1 > 0.85
        assert c1 > c0 + 0.1
        assert abs(p1) < 0.05
    _phase, height = truth_phase(pair, 1, 0, s1sim.LPB)
    elevation = read_map(project, "hill_co_b02_elevation")
    assert height.max() > 500
    assert np.nanmax(np.abs(elevation - height)) < 0.5
    meta = description(project, "hill_co_b02_elevation")
    assert meta["coregistration"]["dem"] == str(dem)
    assert meta["measure"] == "elevation"


def test_debursted(project, pairs, library):
    pair, _dem = pairs["flat"]
    proc = run_module(
        project,
        library,
        reference="deb_ref_iw1_vv",
        secondary="deb_sec_iw1_vv",
        output="deb_co",
    )
    assert proc.returncode == 0, proc.stderr
    assert "ESD" in proc.stderr
    m = read_complex(project, "deb_ref_iw1_vv")
    s = read_complex(project, "deb_co")
    assert m.shape == s.shape
    segments = description(project, "deb_ref_iw1_vv_i")["raster_geometry"]["segments"]
    phase = np.full(m.shape, np.nan)
    for seg in segments:
        p, _h = truth_phase(pair, seg["burst"] - 1, seg["first_line"], seg["rows"])
        phase[seg["first_row"] : seg["first_row"] + seg["rows"]] = p
    # Away from the seams (where the two images hold different bursts for
    # a few rows) the pair is as coherent as with burst maps.
    for seg in segments:
        rows = slice(seg["first_row"] + 4, seg["first_row"] + seg["rows"] - 4)
        coh, _phase = coherence(m, s, phase, rows=rows)
        assert coh > 0.97


def test_metadata(project, library):
    proc = run_module(
        project,
        library,
        reference="flat_ref_iw1_vv",
        secondary="flat_sec_iw1_vv",
        output="meta",
    )
    assert proc.returncode == 0, proc.stderr
    meta = description(project, "meta_b02_q")
    ref = description(project, "flat_ref_iw1_vv_b02_i")
    sec = description(project, "flat_sec_iw1_vv_b02_i")
    assert meta["raster_geometry"] == ref["raster_geometry"]
    assert meta["swath"]["orbit_state_vectors"] == sec["swath"]["orbit_state_vectors"]
    assert meta["product"]["absolute_orbit_start"] == "46927"
    co = meta["coregistration"]
    assert co["reference_maps"] == ["flat_ref_iw1_vv_b02_i", "flat_ref_iw1_vv_b02_q"]
    assert co["burst_pairs"] == [{"reference_burst": 2, "secondary_bursts": [2]}]
    assert co["interpolation"] == "sinc8"
    info = gs.raster_info("meta_b02_q", env=project.env)
    assert info["semantic_label"] == "S1_VV_Q"
    assert info["datatype"] == "FCELL"
    assert int(info["rows"]) == s1sim.LPB


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"reference": "flat_ref_iw1_vv", "secondary": "deb_sec_iw1_vv"}, "both"),
        ({"reference": "flat_ref_iw1_vv", "secondary": "nothing"}, "No complex maps"),
        (
            {
                "reference": "flat_ref_iw1_vv",
                "secondary": "flat_sec_iw1_vv",
                "device": "gpu",
                "platform": "none-such",
            },
            "unavailable",
        ),
    ],
)
def test_failures(project, library, kwargs, message):
    proc = run_module(project, library, output="fail", **kwargs)
    assert proc.returncode != 0
    assert message in proc.stderr


def test_no_overwrite(project, library):
    kwargs = {
        "reference": "flat_ref_iw1_vv",
        "secondary": "flat_sec_iw1_vv",
        "output": "again",
    }
    assert run_module(project, library, **kwargs).returncode == 0
    proc = run_module(project, library, **kwargs)
    assert proc.returncode != 0
    assert "already exists" in proc.stderr
