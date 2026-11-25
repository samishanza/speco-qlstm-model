"""Data for the SPECO QLSTM: fetch, feature engineering, chronological split and sequence windows.

Source data (see README for licences):
- Methane, default: NOAA GML CarbonTracker-CH4 (CT-CH4-2025), lowest model level CH4 mole fraction, 3 deg lon x
  2 deg lat, read from the open NOAA archive (no account) and averaged to monthly means.
- Methane, alternative: CAMS global greenhouse gas reanalysis (EGG4), monthly column-averaged CH4 (Copernicus ADS;
  needs an account and accepted ADS terms).
- Weather: ERA5 single-level monthly means (Copernicus CDS): 10 m wind, 2 m temperature, surface pressure.
- Hotspot selection: EDGAR v8.0 CH4 total emissions, 2022, 0.1 deg (JRC).

Output schema of the fetch (one row per cell per month):
    cell_id, lat, lon, date, ch4, wind_speed, temperature, pressure, edgar_emission
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DEFAULT_CSV = ROOT / "data" / "conus_monthly.csv"
EDGAR_DEFAULT = Path.home() / "topoflow" / "data" / "cache" / "edgar" / "edgar_v8_ch4_2022_0p1.nc"

CONUS_AREA = [50, -125, 24, -66]  # N, W, S, E (CDS/ADS order)
GRID = 0.75
ADS_URL = "https://ads.atmosphere.copernicus.eu/api"
CDS_URL = "https://cds.climate.copernicus.eu/api"

WEATHER = ["wind_speed", "temperature", "pressure"]
TARGET = "ch4"


# ---------------------------------------------------------------------------------------------------------------
# Fetch (Copernicus ADS + CDS)
# ---------------------------------------------------------------------------------------------------------------

def _cds_key() -> str:
    """Read the personal access token from ~/.cdsapirc without printing it."""
    for line in (Path.home() / ".cdsapirc").read_text().splitlines():
        if line.strip().startswith("key:"):
            return line.split(":", 1)[1].strip()
    raise RuntimeError("no key in ~/.cdsapirc")


def _client(url: str):
    import cdsapi
    return cdsapi.Client(url=url, key=_cds_key(), quiet=True, progress=False)


def _check_disk(min_gb: float = 2.0) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(RAW).free / 1e9
    if free < min_gb:
        raise RuntimeError(f"only {free:.1f} GB free; refusing to download")


def download_egg4(years: list[int], target: Path) -> Path:
    """CAMS EGG4 monthly-mean column-averaged CH4 over CONUS."""
    if target.exists():
        return target
    _check_disk()
    part = target.with_suffix(".part")
    _client(ADS_URL).retrieve("cams-global-ghg-reanalysis-egg4-monthly", {
        "variable": ["ch4_column_mean_molar_fraction"],
        "product_type": ["monthly_mean"],
        "year": [str(y) for y in years],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "area": CONUS_AREA,
        "data_format": "netcdf",
    }, str(part))
    part.rename(target)
    return target


def download_era5(years: list[int], target: Path) -> Path:
    """ERA5 monthly means of 10 m wind, 2 m temperature and surface pressure over CONUS on the 0.75 deg grid."""
    if target.exists():
        return target
    _check_disk()
    part = target.with_suffix(".part")
    _client(CDS_URL).retrieve("reanalysis-era5-single-levels-monthly-means", {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind", "2m_temperature", "surface_pressure"],
        "year": [str(y) for y in years],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "time": ["00:00"],
        "area": CONUS_AREA,
        "grid": [GRID, GRID],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }, str(part))
    part.rename(target)
    return target


def _open(path: Path):
    """Open a NetCDF download, unzipping first if the service returned a zip."""
    import xarray as xr
    import zipfile
    if zipfile.is_zipfile(path):
        out = path.with_suffix("")
        out.mkdir(exist_ok=True)
        with zipfile.ZipFile(path) as z:
            z.extractall(out)
        files = sorted(out.glob("*.nc"))
        return xr.open_mfdataset(files, combine="by_coords") if len(files) > 1 else xr.open_dataset(files[0])
    return xr.open_dataset(path)


def xr_open(path: Path):
    import xarray as xr
    return xr.open_dataset(path)


def _std(ds):
    """Normalise coordinate names to lat/lon/time and longitudes to -180..180."""
    ren = {}
    for a, b in (("latitude", "lat"), ("longitude", "lon"), ("valid_time", "time")):
        if a in ds.coords or a in ds.dims:
            ren[a] = b
    ds = ds.rename(ren)
    if float(ds.lon.max()) > 180:
        ds = ds.assign_coords(lon=((ds.lon + 180) % 360) - 180).sortby("lon")
    return ds.sortby("lat")


def edgar_on_grid(lats: np.ndarray, lons: np.ndarray, path: Path = EDGAR_DEFAULT,
                  dlat: float = GRID, dlon: float = GRID) -> np.ndarray:
    """Sum EDGAR 0.1 deg emissions (tonnes CH4/yr) into dlat x dlon cells centred on (lats, lons)."""
    import xarray as xr
    e = xr.open_dataset(path)["emissions"].load()
    out = np.zeros((len(lats), len(lons)))
    for i, la in enumerate(lats):
        band = e.sel(lat=slice(la - dlat / 2, la + dlat / 2))
        for j, lo in enumerate(lons):
            out[i, j] = float(band.sel(lon=slice(lo - dlon / 2, lo + dlon / 2)).sum())
    return out


def pick_cells(emis: np.ndarray, valid: np.ndarray, n_cells: int) -> list[tuple[int, int]]:
    """Indices of the n_cells largest-emission grid cells among those where `valid` is True."""
    picked = []
    for k in np.argsort(emis.ravel(), kind="stable")[::-1]:
        i, j = np.unravel_index(k, emis.shape)
        if valid[i, j]:
            picked.append((int(i), int(j)))
        if len(picked) == n_cells:
            break
    return picked


def tidy_rows(picked, lats, lons, dates, ch4, speed, t2m_k, sp_pa, emis) -> pd.DataFrame:
    """Long-format table in the fetch output schema from (time, lat, lon) arrays on a common grid."""
    dates = pd.to_datetime(dates).to_period("M").to_timestamp()
    rows = []
    for cid, (i, j) in enumerate(picked):
        rows.append(pd.DataFrame({
            "cell_id": cid, "lat": float(lats[i]), "lon": float(lons[j]), "date": dates,
            "ch4": ch4[:, i, j], "wind_speed": speed[:, i, j], "temperature": t2m_k[:, i, j] - 273.15,
            "pressure": sp_pa[:, i, j] / 100.0, "edgar_emission": emis[i, j],
        }))
    out = pd.concat(rows, ignore_index=True).dropna(subset=["ch4"])
    return out[["cell_id", "lat", "lon", "date", "ch4", "wind_speed", "temperature", "pressure", "edgar_emission"]]


def fetch_copernicus(out_csv: Path = DEFAULT_CSV, start_year: int = 2003, end_year: int = 2020,
                     n_cells: int = 40, edgar_path: Path = EDGAR_DEFAULT) -> pd.DataFrame:
    """Download CAMS EGG4 CH4 + ERA5 weather for CONUS, pick the top-N EDGAR cells, write the tidy CSV."""
    years = list(range(start_year, end_year + 1))
    tag = f"{start_year}-{end_year}"
    ch4 = _std(_open(download_egg4(years, RAW / f"egg4_xch4_conus_{tag}.nc")))
    met = _std(_open(download_era5(years, RAW / f"era5_monthly_conus_{tag}.nc")))

    ch4_var = [v for v in ch4.data_vars if "ch4" in v.lower() or "xch4" in v.lower()][0]
    xch4 = ch4[ch4_var]
    if "step" in xch4.dims:
        xch4 = xch4.isel(step=0)
    xch4 = xch4.interp(lat=met.lat, lon=met.lon)  # EGG4 and ERA5 both 0.75 deg; interp aligns any offset
    if float(xch4.max()) < 1e-3:  # mole fraction -> ppb
        xch4 = xch4 * 1e9

    lats, lons = met.lat.values, met.lon.values
    emis = edgar_on_grid(lats, lons, edgar_path)
    flat = np.argsort(emis.ravel())[::-1]
    picked = []
    for k in flat:
        i, j = np.unravel_index(k, emis.shape)
        if np.isfinite(xch4.isel(lat=i, lon=j).values).all():
            picked.append((i, j))
        if len(picked) == n_cells:
            break

    u, v = met["u10"], met["v10"]
    speed = np.sqrt(u ** 2 + v ** 2)
    rows = []
    for cid, (i, j) in enumerate(picked):
        t = pd.to_datetime(met.time.values).to_period("M").to_timestamp()
        c = pd.Series(xch4.isel(lat=i, lon=j).values, index=pd.to_datetime(xch4.time.values).to_period("M").to_timestamp())
        df = pd.DataFrame({
            "cell_id": cid, "lat": float(lats[i]), "lon": float(lons[j]), "date": t,
            "wind_speed": speed.isel(lat=i, lon=j).values,
            "temperature": met["t2m"].isel(lat=i, lon=j).values - 273.15,
            "pressure": met["sp"].isel(lat=i, lon=j).values / 100.0,
            "edgar_emission": emis[i, j],
        })
        df["ch4"] = df["date"].map(c)
        rows.append(df)
    out = pd.concat(rows).dropna(subset=["ch4"])
    out = out[["cell_id", "lat", "lon", "date", "ch4", "wind_speed", "temperature", "pressure", "edgar_emission"]]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out


# ---------------------------------------------------------------------------------------------------------------
# Fetch (NOAA GML CarbonTracker-CH4, open archive, no account)
# ---------------------------------------------------------------------------------------------------------------
# CT-CH4-2025 publishes one file per day (8 three-hourly means, 25 levels, global 3 x 2 deg; ~36 MB each). Only
# the lowest model level over CONUS is needed (~100 kB per day), so each file is read with a single HTTP multi-range
# request for exactly those bytes instead of downloading it. The byte layout below was read from the files
# themselves (netCDF4/HDF5, contiguous little-endian float64 `ch4[time, level, latitude, longitude]`); every response
# is checked against the expected file size, and `verify_ctch4_layout` compares the range read with a regular
# netCDF read of the same file (done for 1 January of every year fetched). The release directory also holds older
# `CTCH4_2024.*` copies for 1998-2016 with identical CH4 values; the `CTCH4_2025.*` files cover every year.

CT_RELEASE = "CT-CH4-2025"
CT_BASE = "https://gml.noaa.gov/aftp/products/carbontracker/ch4/CT-CH4-2025/molefractions"
CT_YEARS = (1998, 2023)
CT_LATS = -89.0 + 2.0 * np.arange(90)     # cell centres; cells are 2 deg in latitude
CT_LONS = -178.5 + 3.0 * np.arange(120)   # cells are 3 deg in longitude
CT_DLAT, CT_DLON = 2.0, 3.0
CT_SHAPE = (8, 25, 90, 120)               # time (3-hourly), level, latitude, longitude
CT_CH4_OFFSET = 21132                     # byte offset of the ch4 variable's data (CTCH4_2025.* files)
CT_FILE_SIZE = 35973664
# CONUS box lat 24..50, lon -125..-66 -> CT cell centres lat 25..49, lon -124.5..-67.5
CT_LAT_IDX = np.where((CT_LATS > 24) & (CT_LATS < 50))[0]
CT_LON_IDX = np.where((CT_LONS > -125) & (CT_LONS < -66))[0]


def ctch4_url(day: pd.Timestamp) -> str:
    return f"{CT_BASE}/{day:%Y}/{day:%m}/CTCH4_2025.molefrac_glb3x2_{day:%Y-%m-%d}.nc"


def ctch4_byte_ranges(level: int = 0, lat_idx: np.ndarray = CT_LAT_IDX) -> list[tuple[int, int]]:
    """Inclusive byte ranges holding ch4[t, level, lat_idx[0]..lat_idx[-1], :] for each of the 8 time steps."""
    nt, nlev, nlat, nlon = CT_SHAPE
    row = nlon * 8
    out = []
    for t in range(nt):
        start = CT_CH4_OFFSET + ((t * nlev + level) * nlat + int(lat_idx[0])) * row
        out.append((start, start + len(lat_idx) * row - 1))
    return out


def parse_multipart_ranges(body: bytes, content_type: str) -> dict[int, tuple[int, bytes]]:
    """Parse a multipart/byteranges HTTP body into {range_start: (total_file_size, payload)}."""
    boundary = content_type.split("boundary=", 1)[1].strip().strip('"').encode()
    parts = {}
    for chunk in body.split(b"--" + boundary):
        if b"\r\n\r\n" not in chunk:
            continue
        head, payload = chunk.split(b"\r\n\r\n", 1)
        rng = [ln for ln in head.decode("latin-1").split("\r\n") if ln.lower().startswith("content-range:")]
        if not rng:
            continue
        spec = rng[0].split(":", 1)[1].strip().split()[1]          # "a-b/total"
        ab, total = spec.split("/")
        a, b = (int(x) for x in ab.split("-"))
        parts[a] = (int(total), payload[:b - a + 1])
    return parts


def ctch4_day_surface(day: pd.Timestamp, session=None, retries: int = 4) -> np.ndarray:
    """CT-CH4 lowest-level CH4 (ppb) over CONUS for one day: array (8 three-hourly steps, lat, lon)."""
    import time
    import requests
    s = session or requests.Session()
    ranges = ctch4_byte_ranges()
    hdr = {"Range": "bytes=" + ",".join(f"{a}-{b}" for a, b in ranges)}
    for attempt in range(retries):
        try:
            r = s.get(ctch4_url(day), headers=hdr, timeout=60)
            if r.status_code != 206 or "multipart/byteranges" not in r.headers.get("Content-Type", ""):
                raise RuntimeError(f"HTTP {r.status_code} {r.headers.get('Content-Type')}")
            parts = parse_multipart_ranges(r.content, r.headers["Content-Type"])
            break
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    out = np.empty((len(ranges), len(CT_LAT_IDX), CT_SHAPE[3]))
    for t, (a, b) in enumerate(ranges):
        total, payload = parts[a]
        if total != CT_FILE_SIZE or len(payload) != b - a + 1:
            raise RuntimeError(f"{ctch4_url(day)}: unexpected file layout (size {total}); run verify_ctch4_layout")
        out[t] = np.frombuffer(payload, "<f8").reshape(len(CT_LAT_IDX), CT_SHAPE[3])
    return out[:, :, CT_LON_IDX]


def verify_ctch4_layout(day: pd.Timestamp) -> float:
    """Max |difference| between the byte-range read and a netCDF read of the same remote file (should be 0)."""
    import netCDF4
    with netCDF4.Dataset(ctch4_url(day) + "#mode=bytes") as d:
        ref = np.asarray(d["ch4"][:, 0, CT_LAT_IDX[0]:CT_LAT_IDX[-1] + 1, CT_LON_IDX[0]:CT_LON_IDX[-1] + 1])
        assert np.allclose(d["latitude"][CT_LAT_IDX], CT_LATS[CT_LAT_IDX])
        assert np.allclose(d["longitude"][CT_LON_IDX], CT_LONS[CT_LON_IDX])
    return float(np.abs(ctch4_day_surface(day) - ref).max())


def monthly_mean_of_days(days: dict[pd.Timestamp, np.ndarray]) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Average per-day arrays (steps, lat, lon) into calendar-month means; every day must be present."""
    keys = sorted(days)
    months = pd.DatetimeIndex(keys).to_period("M").unique()
    out = []
    for m in months:
        want = pd.date_range(m.start_time, m.end_time.normalize(), freq="D")
        missing = [d for d in want if d not in days]
        if missing:
            raise ValueError(f"{m}: {len(missing)} missing days, e.g. {missing[0]:%Y-%m-%d}")
        out.append(np.concatenate([days[d] for d in want]).mean(0))
    return months.to_timestamp(), np.stack(out)


def download_ctch4_monthly(years: list[int], target: Path, threads: int = 8, log=print) -> Path:
    """Monthly-mean CT-CH4 lowest-level CH4 over CONUS (ppb), saved as NetCDF. Resumable per month."""
    import requests
    import xarray as xr
    from concurrent.futures import ThreadPoolExecutor
    if target.exists():
        return target
    if years[0] < CT_YEARS[0] or years[-1] > CT_YEARS[1]:
        raise ValueError(f"{CT_RELEASE} covers {CT_YEARS[0]}-{CT_YEARS[1]}")
    _check_disk()
    cache = target.parent / (target.stem + "_months")
    cache.mkdir(parents=True, exist_ok=True)
    sessions: dict[int, requests.Session] = {}

    def day(d):
        import threading
        s = sessions.setdefault(threading.get_ident(), requests.Session())
        return d, ctch4_day_surface(d, s)

    with ThreadPoolExecutor(threads) as pool:
        diffs = [verify_ctch4_layout(pd.Timestamp(y, 1, 1)) for y in years]  # serial: netCDF-C isn't thread-safe
        if max(diffs) != 0:
            raise RuntimeError(f"byte-range reads differ from netCDF reads (max {max(diffs)} ppb)")
        log(f"  CT-CH4 byte layout verified against netCDF reads for {len(years)} files")
        for y in years:
            for mo in range(1, 13):
                f = cache / f"{y}-{mo:02d}.npy"
                if f.exists():
                    continue
                d0 = pd.Timestamp(y, mo, 1)
                got = dict(pool.map(day, pd.date_range(d0, d0 + pd.offsets.MonthEnd(0), freq="D")))
                _, arr = monthly_mean_of_days(got)
                np.save(f, arr[0])
            log(f"  CT-CH4 {y} done")
    times = pd.DatetimeIndex([pd.Timestamp(y, m, 1) for y in years for m in range(1, 13)])
    data = np.stack([np.load(cache / f"{t:%Y-%m}.npy") for t in times])
    da = xr.DataArray(data, dims=("time", "lat", "lon"),
                      coords={"time": times, "lat": CT_LATS[CT_LAT_IDX], "lon": CT_LONS[CT_LON_IDX]},
                      name="ch4", attrs={"units": "ppb", "long_name": "CH4 mole fraction, lowest model level, "
                                         "monthly mean of 3-hourly means", "source": CT_RELEASE})
    da.to_dataset().to_netcdf(target)
    return target


def block_mean(da, lats: np.ndarray, lons: np.ndarray, dlat: float, dlon: float):
    """Average a finer lat/lon field into cells centred on (lats, lons) of size dlat x dlon (half-open bounds)."""
    import xarray as xr
    la, lo = da["lat"].values, da["lon"].values
    out = np.full(da.shape[:-2] + (len(lats), len(lons)), np.nan)
    for i, c in enumerate(lats):
        mi = (la >= c - dlat / 2) & (la < c + dlat / 2)
        for j, d in enumerate(lons):
            mj = (lo >= d - dlon / 2) & (lo < d + dlon / 2)
            if mi.any() and mj.any():
                out[..., i, j] = da.values[..., mi, :][..., mj].mean(axis=(-2, -1))
    dims = da.dims[:-2] + ("lat", "lon")
    coords = {k: da[k] for k in da.dims[:-2]} | {"lat": lats, "lon": lons}
    return xr.DataArray(out, dims=dims, coords=coords, name=da.name)


# NOAA PSL NCEP/NCAR Reanalysis 1 monthly means (public, no account): 2.5 deg global grid, sigma=0.995 (lowest
# model sigma level, ~40 m above ground) wind and air temperature, and surface pressure.
NCEP_BASE = "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis.derived/surface"
NCEP_FILES = {"uwnd": "uwnd.sig995.mon.mean.nc", "vwnd": "vwnd.sig995.mon.mean.nc",
              "air": "air.sig995.mon.mean.nc", "pres": "pres.sfc.mon.mean.nc"}


def download_ncep_monthly(years: list[int], target: Path, log=print) -> Path:
    """NCEP R1 monthly u, v (sig995), air temperature (sig995) and surface pressure, subset to CONUS (+ margin).

    Each global file (25-31 MB) is downloaded, subset and deleted; only the small CONUS subset is kept.
    """
    import requests
    import xarray as xr
    if target.exists():
        return target
    _check_disk()
    out = {}
    for var, name in NCEP_FILES.items():
        raw = target.parent / f"ncep_{name}"
        with requests.get(f"{NCEP_BASE}/{name}", stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(raw, "wb") as fh:
                for block in r.iter_content(1 << 20):
                    fh.write(block)
        with xr.open_dataset(raw) as ds:
            da = _std(ds)[var].sel(time=slice(f"{years[0]}-01-01", f"{years[-1]}-12-31"),
                                  lat=slice(20, 55), lon=slice(-130, -60)).load()
        raw.unlink()
        out[var] = da
        log(f"  NCEP R1 {name}: {dict(da.sizes)}")
    xr.Dataset(out).to_netcdf(target)
    return target


def overlap_weights(src_centres: np.ndarray, src_width: float, dst_centres: np.ndarray, dst_width: float,
                    coslat: bool = False) -> np.ndarray:
    """Matrix W (dst, src): fraction of each destination cell covered by each source cell (rows sum to 1).

    With coslat=True the 1-D overlaps are latitude bands weighted by area (difference of sin(lat)).
    """
    w = np.zeros((len(dst_centres), len(src_centres)))
    for i, d in enumerate(dst_centres):
        d0, d1 = d - dst_width / 2, d + dst_width / 2
        for j, c in enumerate(src_centres):
            a, b = max(d0, c - src_width / 2), min(d1, c + src_width / 2)
            if b > a:
                w[i, j] = (np.sin(np.radians(b)) - np.sin(np.radians(a))) if coslat else (b - a)
    tot = w.sum(1, keepdims=True)
    if (tot == 0).any():
        raise ValueError("destination cell not covered by the source grid")
    return w / tot


def area_weighted_regrid(da, lats: np.ndarray, lons: np.ndarray, dlat: float, dlon: float) -> np.ndarray:
    """Conservative (area-overlap-weighted) average of a regular lat/lon field (time, lat, lon) onto target cells."""
    slat, slon = da["lat"].values, da["lon"].values
    wl = overlap_weights(slat, abs(float(np.diff(slat).mean())), lats, dlat, coslat=True)
    wo = overlap_weights(slon, abs(float(np.diff(slon).mean())), lons, dlon)
    return np.einsum("il,tlm,jm->tij", wl, da.transpose("time", "lat", "lon").values, wo)


def fetch_open_methane(out_csv: Path = DEFAULT_CSV, start_year: int = 2003, end_year: int = 2020,
                       n_cells: int = 40, edgar_path: Path = EDGAR_DEFAULT, threads: int = 8,
                       weather: str = "ncep") -> pd.DataFrame:
    """NOAA CT-CH4 (open) + weather on CT-CH4's native 3 x 2 deg grid, top-N EDGAR cells, tidy CSV.

    weather="ncep": NCEP/NCAR R1 monthly means (open), area-weighted onto the CT cells.
    weather="era5": ERA5 monthly means (Copernicus CDS), 0.75 deg points block-averaged into the CT cells.
    """
    years = list(range(start_year, end_year + 1))
    tag = f"{start_year}-{end_year}"
    ch4 = _std(_open(download_ctch4_monthly(years, RAW / f"ctch4_2025_sfc_conus_monthly_{tag}.nc", threads)))["ch4"]
    lats, lons = ch4.lat.values, ch4.lon.values
    ch4_t = pd.to_datetime(ch4.time.values).to_period("M").to_timestamp()
    if weather == "ncep":
        met = _std(xr_open(download_ncep_monthly(years, RAW / f"ncep_r1_monthly_conus_{tag}.nc")))
        met_t = pd.to_datetime(met.time.values).to_period("M").to_timestamp()
        if not met_t.equals(ch4_t):
            raise ValueError("NCEP and CT-CH4 months differ")
        u, v = (area_weighted_regrid(met[k], lats, lons, CT_DLAT, CT_DLON) for k in ("uwnd", "vwnd"))
        t = area_weighted_regrid(met["air"], lats, lons, CT_DLAT, CT_DLON)
        if np.nanmedian(t) < 100:  # degC -> K (tidy_rows expects K)
            t = t + 273.15
        sp = area_weighted_regrid(met["pres"], lats, lons, CT_DLAT, CT_DLON)
        if np.nanmedian(sp) < 2000:  # hPa/mbar -> Pa
            sp = sp * 100.0
        regrid = {"speed": np.sqrt(u ** 2 + v ** 2), "t2m": t, "sp": sp}
        emis = edgar_on_grid(lats, lons, edgar_path, CT_DLAT, CT_DLON)
        valid = np.isfinite(ch4.values).all(0) & np.isfinite(regrid["speed"]).all(0)
        picked = pick_cells(emis, valid, n_cells)
        out = tidy_rows(picked, lats, lons, ch4_t, ch4.values, regrid["speed"], regrid["t2m"], regrid["sp"], emis)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_csv, index=False)
        return out
    met = _std(_open(download_era5(years, RAW / f"era5_monthly_conus_{tag}.nc")))
    speed = np.sqrt(met["u10"] ** 2 + met["v10"] ** 2).rename("speed")
    met_t = pd.to_datetime(met.time.values).to_period("M").to_timestamp()
    if not met_t.equals(ch4_t):
        raise ValueError("ERA5 and CT-CH4 months differ")
    def tll(v):  # drop singleton extras (e.g. ensemble `number`) and order dims as time, lat, lon
        extra = [d for d in v.dims if d not in ("time", "lat", "lon")]
        return v.isel({d: 0 for d in extra}).transpose("time", "lat", "lon")

    regrid = {k: block_mean(tll(v), lats, lons, CT_DLAT, CT_DLON).values
              for k, v in (("speed", speed), ("t2m", met["t2m"]), ("sp", met["sp"]))}
    emis = edgar_on_grid(lats, lons, edgar_path, CT_DLAT, CT_DLON)
    valid = np.isfinite(ch4.values).all(0) & np.isfinite(regrid["speed"]).all(0)
    picked = pick_cells(emis, valid, n_cells)
    out = tidy_rows(picked, lats, lons, ch4_t, ch4.values, regrid["speed"], regrid["t2m"], regrid["sp"], emis)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out


# ---------------------------------------------------------------------------------------------------------------
# Synthetic data (smoke tests only)
# ---------------------------------------------------------------------------------------------------------------

def synthetic_dataset(n_cells: int = 6, n_months: int = 96, seed: int = 0) -> pd.DataFrame:
    """SYNTHETIC data with the real schema, for tests only. Not methane observations."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2012-01-01", periods=n_months, freq="MS")
    t = np.arange(n_months)
    rows = []
    for c in range(n_cells):
        temp = 12 + 10 * np.sin(2 * np.pi * (t - 3) / 12) + rng.normal(0, 1, n_months)
        wind = 4 + np.cos(2 * np.pi * t / 12) + rng.normal(0, 0.3, n_months)
        pres = 1000 + rng.normal(0, 3, n_months)
        ch4 = 1850 + 0.6 * t + 8 * np.sin(2 * np.pi * (t - 8) / 12) - 1.5 * wind + 5 * c + rng.normal(0, 2, n_months)
        rows.append(pd.DataFrame({"cell_id": c, "lat": 30 + c, "lon": -100 + c, "date": dates, "ch4": ch4,
                                  "wind_speed": wind, "temperature": temp, "pressure": pres,
                                  "edgar_emission": 1000.0 * (c + 1)}))
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------------------------------------------
# Features, split, sequences
# ---------------------------------------------------------------------------------------------------------------

def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Per-cell lag/rolling features (from the original drafts), seasonality and location encodings."""
    df = df.sort_values(["cell_id", "date"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    g = df.groupby("cell_id")[TARGET]
    df["ch4_lag1"] = g.shift(1)
    df["ch4_lag2"] = g.shift(2)
    # rolling stats use only past values (shifted), so the current target never feeds its own feature
    df["ch4_roll_mean6"] = g.transform(lambda s: s.shift(1).rolling(6).mean())
    df["ch4_roll_std6"] = g.transform(lambda s: s.shift(1).rolling(6).std())
    m = df["date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * m / 12)
    df["month_cos"] = np.cos(2 * np.pi * m / 12)
    df["lat_n"] = (df["lat"] - 24) / 26
    df["lon_n"] = (df["lon"] + 125) / 59
    return df.dropna().reset_index(drop=True)


FEATURES = [TARGET, *WEATHER, "ch4_lag1", "ch4_lag2", "ch4_roll_mean6", "ch4_roll_std6",
            "month_sin", "month_cos", "lat_n", "lon_n"]
SCALED = [TARGET, *WEATHER, "ch4_lag1", "ch4_lag2", "ch4_roll_mean6", "ch4_roll_std6"]


@dataclass
class Scaler:
    """Min-max scaler fit on training rows only."""
    lo: pd.Series
    hi: pd.Series

    @classmethod
    def fit(cls, df: pd.DataFrame) -> "Scaler":
        return cls(df[SCALED].min(), df[SCALED].max())

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[SCALED] = (df[SCALED] - self.lo) / (self.hi - self.lo).replace(0, 1)
        return df

    def inverse_target(self, y: np.ndarray) -> np.ndarray:
        return y * (self.hi[TARGET] - self.lo[TARGET]) + self.lo[TARGET]


def split(df: pd.DataFrame, train_frac: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split on a common date cut-off, so no test month precedes a training month."""
    dates = np.sort(df["date"].unique())
    cut = dates[int(len(dates) * train_frac)]
    return df[df["date"] < cut].copy(), df[df["date"] >= cut].copy()


def create_sequences(df: pd.DataFrame, seq_len: int = 12, horizon: int = 6,
                     context: pd.DataFrame | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Windows per cell: X = seq_len months of FEATURES, y = the next `horizon` months of CH4.

    `context` (e.g. the end of the training set) may supply input history for the first test windows; targets
    always come from `df` only.
    Returns X (n, seq_len, n_features), y (n, horizon), cell ids (n,).
    """
    xs, ys, cells = [], [], []
    for cid, d in df.groupby("cell_id"):
        d = d.sort_values("date")
        hist = d
        n_ctx = 0
        if context is not None:
            c = context[context["cell_id"] == cid].sort_values("date").tail(seq_len)
            hist = pd.concat([c, d])
            n_ctx = len(c)
        f = hist[FEATURES].to_numpy(np.float32)
        y = hist[TARGET].to_numpy(np.float32)
        for i in range(len(hist) - seq_len - horizon + 1):
            if i + seq_len < n_ctx:  # target would start inside the context (training) rows
                continue
            xs.append(f[i:i + seq_len])
            ys.append(y[i + seq_len:i + seq_len + horizon])
            cells.append(cid)
    if not xs:
        return (np.zeros((0, seq_len, len(FEATURES)), np.float32), np.zeros((0, horizon), np.float32),
                np.zeros(0, int))
    return np.stack(xs), np.stack(ys), np.array(cells)


def load(source: str, csv: Path = DEFAULT_CSV) -> pd.DataFrame:
    if source == "synthetic":
        return synthetic_dataset()
    if not Path(csv).exists():
        raise FileNotFoundError(f"{csv} not found; run `python -m speco_qlstm.fetch` first")
    return pd.read_csv(csv, parse_dates=["date"])

