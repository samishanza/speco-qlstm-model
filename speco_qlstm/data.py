"""Data for the SPECO QLSTM: fetch, feature engineering, chronological split and sequence windows.

Source data (see README for licences):
- Methane: CAMS global greenhouse gas reanalysis (EGG4), monthly means, column-averaged CH4 (Copernicus ADS).
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


def edgar_on_grid(lats: np.ndarray, lons: np.ndarray, path: Path = EDGAR_DEFAULT) -> np.ndarray:
    """Sum EDGAR 0.1 deg emissions (tonnes CH4/yr) into GRID-sized cells centred on (lats, lons)."""
    import xarray as xr
    e = xr.open_dataset(path)["emissions"].load()
    out = np.zeros((len(lats), len(lons)))
    half = GRID / 2
    for i, la in enumerate(lats):
        band = e.sel(lat=slice(la - half, la + half))
        for j, lo in enumerate(lons):
            out[i, j] = float(band.sel(lon=slice(lo - half, lo + half)).sum())
    return out


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

