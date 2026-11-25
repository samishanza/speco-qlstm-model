"""Download methane + weather and write data/conus_monthly.csv.

    python -m speco_qlstm.fetch --methane noaa-ct --start 2003 --end 2020 --cells 40      # open, no account
    python -m speco_qlstm.fetch --methane copernicus --start 2003 --end 2020 --cells 40   # needs ADS terms accepted

noaa-ct: NOAA GML CarbonTracker-CH4 (CT-CH4-2025), lowest model level, monthly means, native 3 x 2 deg grid. Weather
on that grid from NCEP/NCAR Reanalysis 1 (--weather ncep, default, open; area-weighted from 2.5 deg) or ERA5
(--weather era5, Copernicus CDS account; 0.75 deg block-averaged).
copernicus: CAMS EGG4 column-averaged CH4 on the 0.75 deg ERA5 grid (needs ERA5 and accepted ADS terms).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .data import DEFAULT_CSV, EDGAR_DEFAULT, fetch_copernicus, fetch_open_methane


def summarize(df) -> str:
    """Short sanity summary of a fetched table (CH4 in ppb)."""
    import numpy as np
    import pandas as pd
    lines = [f"{len(df)} rows, {df.cell_id.nunique()} cells, {df.date.min():%Y-%m}..{df.date.max():%Y-%m}"]
    lines.append(f"ch4 ppb: min {df.ch4.min():.1f}  p5 {df.ch4.quantile(.05):.1f}  median {df.ch4.median():.1f}  "
                 f"p95 {df.ch4.quantile(.95):.1f}  max {df.ch4.max():.1f}")
    months = pd.date_range(df.date.min(), df.date.max(), freq="MS")
    gaps = {c: len(months) - d.date.nunique() for c, d in df.groupby("cell_id")}
    lines.append(f"cells with missing months: {sum(v > 0 for v in gaps.values())}; "
                 f"NaNs: {int(df.isna().sum().sum())}")
    annual = df.groupby(df.date.dt.year).ch4.mean()
    for a, b in ((annual.index.min(), 2007), (2007, annual.index.max())):
        if a < b and a in annual and b in annual:
            lines.append(f"cell-mean trend {a}-{b}: {(annual[b] - annual[a]) / (b - a):+.1f} ppb/yr")
    yrs = np.array(annual.index)
    lines.append("per cell: id lat lon months ch4_min ch4_mean ch4_max trend(ppb/yr, OLS) wind T(C) p(hPa) edgar(t/yr)")
    for c, d in df.groupby("cell_id"):
        yr = (d.date - d.date.min()).dt.days / 365.25
        slope = np.polyfit(yr, d.ch4, 1)[0]
        lines.append(f"  {c:2d} {d.lat.iloc[0]:5.1f} {d.lon.iloc[0]:7.1f} {len(d):4d} {d.ch4.min():7.1f} "
                     f"{d.ch4.mean():7.1f} {d.ch4.max():7.1f} {slope:+5.1f} {d.wind_speed.mean():4.1f} "
                     f"{d.temperature.mean():5.1f} {d.pressure.mean():6.1f} {d.edgar_emission.iloc[0]:9.0f}")
    lines.append("annual means: " + ", ".join(f"{y}:{annual[y]:.0f}" for y in yrs))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methane", choices=["noaa-ct", "copernicus"], default="noaa-ct")
    ap.add_argument("--weather", choices=["ncep", "era5"], default="ncep",
                    help="noaa-ct only: NCEP/NCAR R1 (open, default) or ERA5 (Copernicus CDS account)")
    ap.add_argument("--start", type=int, default=2003)
    ap.add_argument("--end", type=int, default=2020)
    ap.add_argument("--cells", type=int, default=40, help="top-N grid cells by EDGAR 2022 CH4 emissions")
    ap.add_argument("--edgar", type=Path, default=EDGAR_DEFAULT)
    ap.add_argument("--threads", type=int, default=8, help="parallel HTTP requests to NOAA (noaa-ct only)")
    ap.add_argument("--out", type=Path, default=DEFAULT_CSV)
    a = ap.parse_args()
    if a.methane == "noaa-ct":
        df = fetch_open_methane(a.out, a.start, a.end, a.cells, a.edgar, a.threads, a.weather)
    else:
        df = fetch_copernicus(a.out, a.start, a.end, a.cells, a.edgar)
    print(f"wrote {a.out}")
    print(summarize(df))


if __name__ == "__main__":
    main()
