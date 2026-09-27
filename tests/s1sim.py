"""Simulator of a Sentinel-1 IW TOPS SLC pair written as SAFE products.

The scene is a set of point scatterers on the ground (on the ellipsoid, or
on a synthetic hill). Each acquisition has its own analytic orbit (a great
circle in Earth-fixed coordinates, the secondary one translated by a
baseline) and burst timing. The SLC value of a pixel is the sum over the
scatterers of

    a exp(-j 4 pi R / lambda) h_az(line - l) h_rg(sample - x)
      exp(-j (phi_k(line) - phi_k(l)))

where (l, x) is the position of the scatterer in burst k, h are windowed
sinc impulse responses and phi_k the TOPS deramping phase of the burst
(Doppler rate, reference time and Doppler centroid computed from the
annotated FM rate, Doppler centroid and steering rate, as SNAP does). The
geometry used here (zero-Doppler time, slant range, ground points) is
computed independently of the module under test.

The products are small (3 bursts of 64 lines x 96 samples) so that tests
run in seconds, but have realistic Sentinel-1 geometry and timing.
"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

C = 299792458.0
A = 6378137.0
F = 1.0 / 298.257223563
E2 = F * (2.0 - F)

RADAR_FREQUENCY = 5.405000454334350e9
WAVELENGTH = C / RADAR_FREQUENCY
FS = 6.434523812571428e7
DT = 2.055556e-3
STEERING_RATE = 1.590368784  # degrees per second
SPEED = 7590.0
ALTITUDE = 700e3
NBURSTS = 3
LPB = 64
NSAMPLES = 96
CYCLE_LINES = 52.3
INVALID_LINES = 2
INVALID_SAMPLES = 2
FM_POLY = (-2330.0, 449000.0, -78000000.0)
DC_POLY = (30.0, -9000.0, 0.0)
T0 = datetime(2023, 1, 12, 14, 25, 9)
CENTRE = (25.0, 55.0)
HEADING = -12.0
LOOK_GROUND = 490e3
HALF_IRF = 7
BW_AZ = 0.67
BW_RG = 0.88


def geodetic_to_ecef(lat, lon, h):
    lat = np.radians(lat)
    lon = np.radians(lon)
    n = A / np.sqrt(1 - E2 * np.sin(lat) ** 2)
    return np.stack(
        [
            (n + h) * np.cos(lat) * np.cos(lon),
            (n + h) * np.cos(lat) * np.sin(lon),
            (n * (1 - E2) + h) * np.sin(lat),
        ],
        axis=-1,
    )


def ecef_to_geodetic(p):
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    lon = np.arctan2(y, x)
    rho = np.hypot(x, y)
    lat = np.arctan2(z, rho * (1 - E2))
    for _i in range(8):
        n = A / np.sqrt(1 - E2 * np.sin(lat) ** 2)
        h = rho / np.cos(lat) - n
        lat = np.arctan2(z, rho * (1 - E2 * n / (n + h)))
    n = A / np.sqrt(1 - E2 * np.sin(lat) ** 2)
    return np.degrees(lat), np.degrees(lon), rho / np.cos(lat) - n


def hill(lat, lon, height=0.0):
    """Terrain height: a Gaussian hill at the scene centre."""
    if height == 0.0:
        return np.zeros(np.broadcast(lat, lon).shape)
    # Broad enough for gentle slopes (about 10 degrees at most): steep
    # slopes decorrelate long-baseline pairs whatever the processing.
    d2 = ((lat - CENTRE[0] - 0.01) / 0.03) ** 2 + ((lon - CENTRE[1] + 0.01) / 0.03) ** 2
    return height * np.exp(-0.5 * d2)


class TrueOrbit:
    """Great circle through the reference position at t = 0."""

    def __init__(self, baseline=0.0):
        """baseline: offset (meters) of the orbit perpendicular to the
        track and to the line of sight of the scene centre."""
        g = geodetic_to_ecef(np.array(CENTRE[0]), np.array(CENTRE[1]), np.array(0.0))
        up = g / np.linalg.norm(g)
        east = np.array(
            [-np.sin(np.radians(CENTRE[1])), np.cos(np.radians(CENTRE[1])), 0.0]
        )
        north = np.cross(up, east)
        hd = np.radians(HEADING)
        track = np.cos(hd) * north + np.sin(hd) * east
        left = np.cross(up, track)
        s0 = g + LOOK_GROUND * left
        self.radius = np.linalg.norm(g) + ALTITUDE
        s0 = s0 / np.linalg.norm(s0) * self.radius
        self.e1 = s0 / self.radius
        w = np.cross(self.e1, track)
        w /= np.linalg.norm(w)
        self.e2 = np.cross(w, self.e1)
        self.omega = SPEED / self.radius
        normal = np.cross(self.e2, g - s0)
        self.offset = baseline * normal / np.linalg.norm(normal)

    def state(self, t):
        t = np.asarray(t, dtype=np.float64)[..., None]
        c, s = np.cos(self.omega * t), np.sin(self.omega * t)
        pos = self.radius * (c * self.e1 + s * self.e2) + self.offset
        vel = self.radius * self.omega * (-s * self.e1 + c * self.e2)
        return pos, vel

    def zero_doppler(self, p, t0=0.0):
        t = np.full(p.shape[:-1], t0, dtype=np.float64)
        for _i in range(20):
            pos, vel = self.state(t)
            acc = -(self.omega**2) * (pos - self.offset)
            d = p - pos
            f = np.sum(d * vel, -1)
            fp = -np.sum(vel * vel, -1) + np.sum(d * acc, -1)
            t = t - f / fp
        pos, _v = self.state(t)
        return t, np.linalg.norm(p - pos, axis=-1)

    def ground(self, t, rng, terrain=0.0):
        """Ground point at zero-Doppler time t and slant range rng, right
        looking, by bisection on the look angle; terrain is a height or a
        function of (lat, lon)."""
        pos, vel = self.state(t)
        vhat = vel / np.linalg.norm(vel, axis=-1, keepdims=True)
        down = -pos / np.linalg.norm(pos, axis=-1, keepdims=True)
        e1 = down - np.sum(down * vhat, -1)[..., None] * vhat
        e1 /= np.linalg.norm(e1, axis=-1, keepdims=True)
        e2 = np.cross(e1, vhat)
        rng = np.asarray(rng, dtype=np.float64)[..., None]
        lo = np.zeros(pos.shape[:-1])
        hi = np.full(pos.shape[:-1], np.radians(85.0))
        for _i in range(70):
            mid = 0.5 * (lo + hi)
            p = pos + rng * (np.cos(mid)[..., None] * e1 + np.sin(mid)[..., None] * e2)
            lat, lon, h = ecef_to_geodetic(p)
            target = terrain(lat, lon) if callable(terrain) else terrain
            below = h < target
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        return pos + rng * (np.cos(lo)[..., None] * e1 + np.sin(lo)[..., None] * e2)


class Acquisition:
    """Timing, Doppler and orbit of one simulated acquisition."""

    def __init__(
        self,
        start,
        orbit,
        burst_offset_lines=0.0,
        orbit_time_error=0.0,
        range_time_error=0.0,
        absolute_orbit=46752,
    ):
        self.start = start
        self.orbit = orbit
        self.orbit_time_error = orbit_time_error
        # Error (two-way seconds) of the annotated slant range time.
        self.range_time_error = range_time_error
        self.absolute_orbit = absolute_orbit
        centre = -0.5 * (NBURSTS - 1) * CYCLE_LINES * DT - 0.5 * (LPB - 1) * DT
        self.burst_t = (
            centre + (np.arange(NBURSTS) * CYCLE_LINES + burst_offset_lines) * DT
        )
        g = geodetic_to_ecef(np.array(CENTRE[0]), np.array(CENTRE[1]), np.array(0.0))
        _t, r = orbit.zero_doppler(g[None, :])
        self.srt = 2.0 * (r[0] - 0.5 * NSAMPLES * C / (2 * FS)) / C
        tau = self.srt + np.arange(NSAMPLES) / FS
        dt0 = tau - self.srt
        ka = FM_POLY[0] + FM_POLY[1] * dt0 + FM_POLY[2] * dt0**2
        self.fdc = DC_POLY[0] + DC_POLY[1] * dt0 + DC_POLY[2] * dt0**2
        krot = 2 * SPEED * np.radians(STEERING_RATE) / WAVELENGTH
        self.kt = ka * krot / (ka - krot)
        fvp = INVALID_SAMPLES
        self.tref = LPB * DT / 2 + self.fdc[fvp] / ka[fvp] - self.fdc / ka

    def ramp(self, line, col):
        ta = line * DT
        return (
            -np.pi * self.kt[col] * (ta - self.tref[col]) ** 2
            - 2 * np.pi * self.fdc[col] * ta
        )

    def time_text(self, t):
        return (self.start + timedelta(microseconds=round(t * 1e6))).isoformat(
            timespec="microseconds"
        )

    def pixel_ground(self, k, line, sample, terrain=0.0):
        t = self.burst_t[k] + line * DT
        rng = 0.5 * C * (self.srt + sample / FS)
        return self.orbit.ground(t, rng, terrain)

    def image(self, scatterers, amplitudes):
        """Complex bursts (NBURSTS, LPB, NSAMPLES) of the scene."""
        t, r = self.orbit.zero_doppler(scatterers)
        x = (2 * r / C - self.srt) * FS
        phase = np.exp(-4j * np.pi * r / WAVELENGTH) * amplitudes
        out = np.zeros((NBURSTS, LPB, NSAMPLES), dtype=np.complex128)
        offsets = np.arange(-HALF_IRF, HALF_IRF + 1)
        for k in range(NBURSTS):
            line = (t - self.burst_t[k]) / DT
            sel = (
                (line > -HALF_IRF)
                & (line < LPB + HALF_IRF)
                & (x > -HALF_IRF)
                & (x < NSAMPLES + HALF_IRF)
            )
            ls, xs, ps = line[sel], x[sel], phase[sel]
            acc = np.zeros(LPB * NSAMPLES, dtype=np.complex128)
            for dl in offsets:
                rows = np.floor(ls).astype(int) + dl
                wl = irf(rows - ls, BW_AZ)
                for dx in offsets:
                    cols = np.floor(xs).astype(int) + dx
                    ok = (rows >= 0) & (rows < LPB) & (cols >= 0) & (cols < NSAMPLES)
                    if not ok.any():
                        continue
                    rr, cc = rows[ok], cols[ok]
                    ramp = np.exp(-1j * (self.ramp(rr, cc) - self.ramp(ls[ok], cc)))
                    val = ps[ok] * wl[ok] * irf(cc - xs[ok], BW_RG) * ramp
                    idx = rr * NSAMPLES + cc
                    acc += np.bincount(
                        idx, weights=val.real, minlength=acc.size
                    ) + 1j * np.bincount(idx, weights=val.imag, minlength=acc.size)
            out[k] = acc.reshape(LPB, NSAMPLES)
        return out


def irf(d, bandwidth):
    window = np.where(
        np.abs(d) < HALF_IRF, 0.5 + 0.5 * np.cos(np.pi * d / HALF_IRF), 0.0
    )
    return np.sinc(bandwidth * d) * window


def scene(acq, terrain_height=0.0, density=4.0, seed=1):
    """Random scatterers covering the image of acq, with their amplitudes."""
    rng = np.random.default_rng(seed)
    lines = np.array([-10, LPB + 10, -10, LPB + 10], dtype=float)
    samples = np.array([-15, -15, NSAMPLES + 15, NSAMPLES + 15], dtype=float)

    def terrain(lat, lon):
        return hill(lat, lon, terrain_height)

    corners = np.concatenate(
        [
            acq.pixel_ground(np.full(4, k), lines, samples, terrain)
            for k in (0, NBURSTS - 1)
        ]
    )
    lat, lon, _h = ecef_to_geodetic(corners)
    n = int(density * NBURSTS * LPB * NSAMPLES * 1.6)
    slat = rng.uniform(lat.min(), lat.max(), n)
    slon = rng.uniform(lon.min(), lon.max(), n)
    sh = hill(slat, slon, terrain_height)
    amplitudes = (rng.normal(size=n) + 1j * rng.normal(size=n)) * 30.0
    return geodetic_to_ecef(slat, slon, sh), amplitudes


def annotation_xml(acq, stem_swath="IW1"):
    nlines = NBURSTS * LPB
    fvs = (
        [-1] * INVALID_LINES
        + [INVALID_SAMPLES] * (LPB - 2 * INVALID_LINES)
        + [-1] * INVALID_LINES
    )
    lvs = [-1 if v < 0 else NSAMPLES - 1 - INVALID_SAMPLES for v in fvs]
    bursts = ""
    for k in range(NBURSTS):
        bursts += f"""
      <burst>
        <azimuthTime>{acq.time_text(acq.burst_t[k])}</azimuthTime>
        <azimuthAnxTime>{400.0 + acq.burst_t[k]}</azimuthAnxTime>
        <sensingTime>{acq.time_text(acq.burst_t[k] + 1.0)}</sensingTime>
        <byteOffset>{1000 + k * LPB * NSAMPLES * 4}</byteOffset>
        <firstValidSample count="{LPB}">{" ".join(map(str, fvs))}</firstValidSample>
        <lastValidSample count="{LPB}">{" ".join(map(str, lvs))}</lastValidSample>
        <burstId absolute="{1000 + k}">{500 + k}</burstId>
      </burst>"""
    points = ""
    grid_lines = list(range(0, nlines, 16)) + [nlines - 1]
    grid_pixels = [0, 32, 64, NSAMPLES - 1]
    count = 0
    for gl in grid_lines:
        k, line = min(gl // LPB, NBURSTS - 1), gl - min(gl // LPB, NBURSTS - 1) * LPB
        t = acq.burst_t[k] + line * DT
        p = acq.pixel_ground(
            np.full(len(grid_pixels), k),
            np.full(len(grid_pixels), float(line)),
            np.array(grid_pixels, float),
        )
        lat, lon, h = ecef_to_geodetic(p)
        for j, px in enumerate(grid_pixels):
            count += 1
            points += f"""
      <geolocationGridPoint>
        <azimuthTime>{acq.time_text(t)}</azimuthTime>
        <slantRangeTime>{acq.srt + px / FS:.12e}</slantRangeTime>
        <line>{gl}</line>
        <pixel>{px}</pixel>
        <latitude>{lat[j]:.10f}</latitude>
        <longitude>{lon[j]:.10f}</longitude>
        <height>{h[j]:.4f}</height>
        <incidenceAngle>{33.0 + 0.01 * px}</incidenceAngle>
        <elevationAngle>{29.0 + 0.01 * px}</elevationAngle>
      </geolocationGridPoint>"""
    orbits = ""
    osv_t = np.arange(-6.0, 6.01, 1.0)
    pos, vel = acq.orbit.state(osv_t)
    for i, t in enumerate(osv_t):
        orbits += f"""
      <orbit><time>{acq.time_text(t + acq.orbit_time_error)}</time><frame>Earth Fixed</frame>
        <position><x>{pos[i, 0]:.6f}</x><y>{pos[i, 1]:.6f}</y><z>{pos[i, 2]:.6f}</z></position>
        <velocity><x>{vel[i, 0]:.9f}</x><y>{vel[i, 1]:.9f}</y><z>{vel[i, 2]:.9f}</z></velocity></orbit>"""
    first, last = acq.burst_t[0], acq.burst_t[-1] + (LPB - 1) * DT
    fm = " ".join(repr(c) for c in FM_POLY)
    dc = " ".join(repr(c) for c in DC_POLY)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<product>
  <adsHeader>
    <missionId>S1A</missionId>
    <productType>SLC</productType>
    <polarisation>VV</polarisation>
    <mode>IW</mode>
    <swath>{stem_swath}</swath>
    <startTime>{acq.time_text(first)}</startTime>
    <stopTime>{acq.time_text(last)}</stopTime>
    <absoluteOrbitNumber>{acq.absolute_orbit}</absoluteOrbitNumber>
    <missionDataTakeId>367320</missionDataTakeId>
    <imageNumber>004</imageNumber>
  </adsHeader>
  <generalAnnotation>
    <productInformation>
      <pass>Descending</pass>
      <timelinessCategory>Fast-24h</timelinessCategory>
      <platformHeading>{HEADING}</platformHeading>
      <projection>Slant Range</projection>
      <rangeSamplingRate>{FS!r}</rangeSamplingRate>
      <radarFrequency>{RADAR_FREQUENCY!r}</radarFrequency>
      <azimuthSteeringRate>{STEERING_RATE!r}</azimuthSteeringRate>
    </productInformation>
    <downlinkInformationList count="1">
      <downlinkInformation><prf>1717.128973878037</prf></downlinkInformation>
    </downlinkInformationList>
    <orbitList count="{osv_t.size}">{orbits}
    </orbitList>
    <terrainHeightList count="1">
      <terrainHeight><azimuthTime>{acq.time_text(first)}</azimuthTime><value>0.0</value></terrainHeight>
    </terrainHeightList>
    <azimuthFmRateList count="1">
      <azimuthFmRate><azimuthTime>{acq.time_text(0.0)}</azimuthTime><t0>{float(acq.srt)!r}</t0>
        <azimuthFmRatePolynomial count="3">{fm}</azimuthFmRatePolynomial>
      </azimuthFmRate>
    </azimuthFmRateList>
  </generalAnnotation>
  <imageAnnotation>
    <imageInformation>
      <productFirstLineUtcTime>{acq.time_text(first)}</productFirstLineUtcTime>
      <productLastLineUtcTime>{acq.time_text(last)}</productLastLineUtcTime>
      <slantRangeTime>{float(acq.srt + acq.range_time_error)!r}</slantRangeTime>
      <pixelValue>Complex</pixelValue>
      <outputPixels>16 bit Signed Integer</outputPixels>
      <rangePixelSpacing>{float(C / (2 * FS))!r}</rangePixelSpacing>
      <azimuthPixelSpacing>1.399187e+01</azimuthPixelSpacing>
      <azimuthTimeInterval>{DT!r}</azimuthTimeInterval>
      <numberOfSamples>{NSAMPLES}</numberOfSamples>
      <numberOfLines>{nlines}</numberOfLines>
      <incidenceAngleMidSwath>33.97</incidenceAngleMidSwath>
    </imageInformation>
    <processingInformation>
      <dcMethod>Data Analysis</dcMethod>
      <swathProcParamsList count="1"><swathProcParams>
        <swath>{stem_swath}</swath>
        <rangeProcessing><windowType>Hamming</windowType><processingBandwidth>5.6e+07</processingBandwidth></rangeProcessing>
        <azimuthProcessing><windowType>Hamming</windowType><processingBandwidth>327.0</processingBandwidth></azimuthProcessing>
      </swathProcParams></swathProcParamsList>
    </processingInformation>
  </imageAnnotation>
  <dopplerCentroid>
    <dcEstimateList count="1">
      <dcEstimate><azimuthTime>{acq.time_text(0.0)}</azimuthTime><t0>{float(acq.srt)!r}</t0>
        <geometryDcPolynomial count="3">{dc}</geometryDcPolynomial>
        <dataDcPolynomial count="3">{dc}</dataDcPolynomial>
      </dcEstimate>
    </dcEstimateList>
  </dopplerCentroid>
  <swathTiming>
    <linesPerBurst>{LPB}</linesPerBurst>
    <samplesPerBurst>{NSAMPLES}</samplesPerBurst>
    <burstList count="{NBURSTS}">{bursts}
    </burstList>
  </swathTiming>
  <geolocationGrid>
    <geolocationGridPointList count="{count}">{points}
    </geolocationGridPointList>
  </geolocationGrid>
</product>
"""


def manifest(acq):
    first, last = acq.burst_t[0], acq.burst_t[-1] + (LPB - 1) * DT
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<xfdu:XFDU xmlns:xfdu="urn:ccsds:schema:xfdu:1"
  xmlns:safe="http://www.esa.int/safe/sentinel-1.0"
  xmlns:s1sarl1="http://www.esa.int/safe/sentinel-1.0/sentinel-1/sar/level-1">
  <metadataSection>
    <safe:processing name="SLC Processing">
      <safe:facility name="Test"><safe:software name="Sentinel-1 IPF" version="003.52"/></safe:facility>
    </safe:processing>
    <safe:platform>
      <safe:familyName>SENTINEL-1</safe:familyName>
      <safe:number>A</safe:number>
      <safe:instrument><safe:extension><s1sarl1:instrumentMode>
        <s1sarl1:mode>IW</s1sarl1:mode></s1sarl1:instrumentMode></safe:extension></safe:instrument>
    </safe:platform>
    <safe:orbitReference>
      <safe:orbitNumber type="start">{acq.absolute_orbit}</safe:orbitNumber>
      <safe:relativeOrbitNumber type="start">130</safe:relativeOrbitNumber>
      <safe:cycleNumber>279</safe:cycleNumber>
      <safe:extension><s1:orbitProperties xmlns:s1="http://www.esa.int/safe/sentinel-1.0/sentinel-1">
        <s1:pass>DESCENDING</s1:pass></s1:orbitProperties></safe:extension>
    </safe:orbitReference>
    <safe:acquisitionPeriod>
      <safe:startTime>{acq.time_text(first)}</safe:startTime>
      <safe:stopTime>{acq.time_text(last)}</safe:stopTime>
    </safe:acquisitionPeriod>
    <s1sarl1:standAloneProductInformation>
      <s1sarl1:transmitterReceiverPolarisation>VV</s1sarl1:transmitterReceiverPolarisation>
      <s1sarl1:productType>SLC</s1sarl1:productType>
    </s1sarl1:standAloneProductInformation>
  </metadataSection>
</xfdu:XFDU>
"""


def write_safe(parent, acq, bursts):
    """Write the SAFE product of acq with complex bursts; return its path."""
    from osgeo import gdal

    gdal.UseExceptions()
    stamp = (acq.start + timedelta(seconds=float(acq.burst_t[0]))).strftime(
        "%Y%m%dT%H%M%S"
    )
    name = f"S1A_IW_SLC__1SDV_{stamp}_{stamp}_{acq.absolute_orbit:06d}_059AD8_SIM0.SAFE"
    stem = f"s1a-iw1-slc-vv-{stamp.lower()}-{stamp.lower()}-{acq.absolute_orbit:06d}-059ad8-004"
    safe = Path(parent) / name
    for sub in ("annotation", "measurement"):
        (safe / sub).mkdir(parents=True, exist_ok=True)
    (safe / "manifest.safe").write_text(manifest(acq))
    (safe / "annotation" / f"{stem}.xml").write_text(annotation_xml(acq))
    data = bursts.reshape(NBURSTS * LPB, NSAMPLES)
    scale = 200.0 / np.sqrt(np.mean(np.abs(data) ** 2))
    q = np.round(data * scale)
    ds = gdal.GetDriverByName("GTiff").Create(
        str(safe / "measurement" / f"{stem}.tiff"),
        NSAMPLES,
        NBURSTS * LPB,
        1,
        gdal.GDT_CInt16,
    )
    ds.GetRasterBand(1).WriteArray(q.astype(np.complex64))
    ds = None
    return safe


def write_dem(path, terrain_height, bbox, step=0.0002):
    """GeoTIFF of the hill (ellipsoidal heights) in WGS84 lon/lat."""
    from osgeo import gdal, osr

    west, south, east, north = bbox
    lons = np.arange(west, east, step) + step / 2
    lats = np.arange(north, south, -step) - step / 2
    lon, lat = np.meshgrid(lons, lats)
    z = hill(lat, lon, terrain_height).astype(np.float32)
    ds = gdal.GetDriverByName("GTiff").Create(
        str(path), lons.size, lats.size, 1, gdal.GDT_Float32
    )
    ds.SetGeoTransform((west, step, 0, north, 0, -step))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())
    ds.GetRasterBand(1).WriteArray(z)
    ds = None
    return path


class Pair:
    """A simulated reference/secondary pair and its ground truth."""

    def __init__(
        self,
        parent,
        baseline=0.0,
        terrain_height=0.0,
        orbit_time_error=0.0,
        range_time_error=0.0,
        seed=1,
    ):
        self.terrain_height = terrain_height
        self.ref = Acquisition(T0, TrueOrbit())
        self.sec = Acquisition(
            T0 + timedelta(days=12),
            TrueOrbit(baseline),
            burst_offset_lines=2.37,
            orbit_time_error=orbit_time_error,
            range_time_error=range_time_error,
            absolute_orbit=46927,
        )
        points, amplitudes = scene(self.ref, terrain_height, seed=seed)
        self.ref_bursts = self.ref.image(points, amplitudes)
        self.sec_bursts = self.sec.image(points, amplitudes)
        parent = Path(parent)
        self.ref_safe = write_safe(parent / "ref", self.ref, self.ref_bursts)
        self.sec_safe = write_safe(parent / "sec", self.sec, self.sec_bursts)

    def terrain(self, lat, lon):
        return hill(lat, lon, self.terrain_height)

    def bbox(self, margin=0.03):
        p = np.concatenate(
            [
                self.ref.pixel_ground(
                    np.full(2, k),
                    np.array([0.0, LPB - 1.0]),
                    np.array([0.0, NSAMPLES - 1.0]),
                )
                for k in (0, NBURSTS - 1)
            ]
        )
        lat, lon, _h = ecef_to_geodetic(p)
        return (
            lon.min() - margin,
            lat.min() - margin,
            lon.max() + margin,
            lat.max() + margin,
        )

    def reference_phase(self, k, lines, samples, topography=True):
        """True flat-earth (and topographic) phase -4 pi (R_ref - R_sec) /
        lambda of reference burst k pixels, and their terrain height."""
        terrain = self.terrain if topography else 0.0
        p = self.ref.pixel_ground(
            np.full(lines.shape, k), lines.astype(float), samples.astype(float), terrain
        )
        _t, r_ref = self.ref.orbit.zero_doppler(p)
        _t, r_sec = self.sec.orbit.zero_doppler(p)
        return -4 * np.pi * (r_ref - r_sec) / WAVELENGTH, ecef_to_geodetic(p)[2]
