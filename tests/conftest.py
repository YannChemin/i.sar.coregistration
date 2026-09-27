"""Fixtures for i.sar.coregistration tests.

Simulated Sentinel-1 TOPS pairs (s1sim.py) are written as SAFE products,
imported with r.in.s1slc into an XY project, and coregistered with the
script and the compute library built from the source tree.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import grass.script as gs

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent
SCRIPT = str(SOURCE / "i.sar.coregistration.py")
sys.path.insert(0, str(HERE))

import s1sim  # noqa: E402

# Simulated cases: baseline (m), hill height (m), secondary orbit timing
# error (lines), secondary slant range time error (samples).
CASES = {
    "flat": (150.0, 0.0, 0.05, 0.0),
    "hill": (1000.0, 800.0, 0.0, 0.0),
    "range": (150.0, 0.0, 0.0, 0.3),
}


def r_in_s1slc():
    """Command running r.in.s1slc from its source tree or installation."""
    path = os.environ.get(
        "R_IN_S1SLC", str(Path.home() / "dev" / "r.in.s1slc" / "r.in.s1slc.py")
    )
    if os.path.isfile(path):
        return [sys.executable, path]
    if shutil.which("r.in.s1slc"):
        return ["r.in.s1slc"]
    return None


@pytest.fixture(scope="session")
def library(tmp_path_factory):
    """Build libsarcoreg.so with the flags of the Makefile."""
    if not shutil.which("gcc"):
        pytest.skip("gcc not available")
    build = tmp_path_factory.mktemp("build")
    source = (SOURCE / "sarcoreg_core.h").read_text() + (
        SOURCE / "sarcoreg_kernels.cl"
    ).read_text()
    lines = [
        '"' + line.replace("\\", "\\\\").replace('"', '\\"') + '\\n"'
        for line in source.splitlines()
    ]
    (build / "sarcoreg_cl.h").write_text("\n".join(lines) + "\n")
    lib = build / "libsarcoreg.so"
    subprocess.run(
        [
            "gcc",
            "-O3",
            "-std=gnu11",
            "-fPIC",
            "-fopenmp",
            "-shared",
            f"-I{build}",
            f"-I{SOURCE}",
            "-o",
            str(lib),
            str(SOURCE / "sarcoreg.c"),
            "-lOpenCL",
            "-lm",
        ],
        check=True,
    )
    return lib


def opencl_available(library):
    import ctypes

    lib = ctypes.CDLL(str(library))
    msg = ctypes.create_string_buffer(4096)
    ok = (
        lib.sarcoreg_init(b"cpu", b"", 0, msg, 4096) == 0
        or lib.sarcoreg_init(b"gpu", b"", 0, msg, 4096) == 0
    )
    lib.sarcoreg_finish()
    return ok


@pytest.fixture(scope="session")
def pairs(tmp_path_factory):
    command = r_in_s1slc()
    if command is None:
        pytest.skip("r.in.s1slc not available (set R_IN_S1SLC to its script)")
    out = {}
    for name, (baseline, hill, error, range_error) in CASES.items():
        parent = tmp_path_factory.mktemp(name)
        pair = s1sim.Pair(
            parent,
            baseline=baseline,
            terrain_height=hill,
            orbit_time_error=error * s1sim.DT,
            range_time_error=range_error / s1sim.FS,
        )
        dem = None
        if hill:
            dem = s1sim.write_dem(parent / "dem.tif", hill, pair.bbox())
        out[name] = (pair, dem)
    return out


@pytest.fixture(scope="session")
def project(tmp_path_factory, pairs):
    """XY project with the pairs imported as burst maps (flat_ref_iw1_vv,
    ...) and the flat pair also debursted (deb_ref_iw1_vv, ...)."""
    path = tmp_path_factory.mktemp("grassdata") / "xy"
    gs.create_project(path)
    with gs.setup.init(path, env=os.environ.copy()) as session:
        command = r_in_s1slc()
        for name, (pair, _dem) in pairs.items():
            for role, safe in (("ref", pair.ref_safe), ("sec", pair.sec_safe)):
                subprocess.run(
                    [*command, "-b", f"input={safe}", f"output={name}_{role}", "--q"],
                    env=session.env,
                    check=True,
                )
        pair = pairs["flat"][0]
        for role, safe in (("ref", pair.ref_safe), ("sec", pair.sec_safe)):
            subprocess.run(
                [*command, f"input={safe}", f"output=deb_{role}", "--q"],
                env=session.env,
                check=True,
            )
        yield session


def run_module(session, library, *flags, **kwargs):
    args = [sys.executable, SCRIPT]
    args += ["-" + f for f in flags]
    args += [f"{k}={v}" for k, v in kwargs.items()]
    env = dict(session.env)
    env["I_SAR_COREGISTRATION_LIB"] = str(library)
    return subprocess.run(args, env=env, capture_output=True, text=True, check=False)


def read_map(session, name):
    import grass.script.array as garray

    env = dict(session.env)
    env["GRASS_REGION"] = gs.region_env(raster=name, env=env)
    return np.array(
        garray.array(name, null="nan", dtype=np.float32, env=env), dtype=np.float64
    )


def read_complex(session, stem):
    return read_map(session, stem + "_i") + 1j * read_map(session, stem + "_q")


def description(session, name):
    env = gs.gisenv(env=session.env)
    path = os.path.join(
        env["GISDBASE"],
        env["LOCATION_NAME"],
        env["MAPSET"],
        "cell_misc",
        name,
        "description.json",
    )
    with open(path) as fd:
        return json.load(fd)


def truth_phase(pair, burst, first_line, rows):
    """True interferometric phase of reference burst lines, every sample."""
    lines, samples = np.mgrid[first_line : first_line + rows, 0 : s1sim.NSAMPLES]
    phase, height = pair.reference_phase(burst, lines.ravel(), samples.ravel())
    return phase.reshape(lines.shape), height.reshape(lines.shape)


def coherence(m, s, phase, rows=slice(4, -4), cols=slice(6, -6)):
    """Coherence of m s* exp(-j phase) over the given window of valid pixels."""
    ifg = (m * np.conj(s) * np.exp(-1j * phase))[rows, cols]
    valid = np.isfinite(ifg)
    num = np.sum(ifg[valid])
    den = np.sqrt(
        np.sum(np.abs(m[rows, cols][valid]) ** 2)
        * np.sum(np.abs(s[rows, cols][valid]) ** 2)
    )
    return float(np.abs(num) / den), float(np.angle(num))
