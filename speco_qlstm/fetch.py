"""Download CAMS EGG4 methane + ERA5 weather from Copernicus and write data/conus_monthly.csv.

    python -m speco_qlstm.fetch --start 2003 --end 2020 --cells 40
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .data import DEFAULT_CSV, EDGAR_DEFAULT, fetch_copernicus


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=int, default=2003)
    ap.add_argument("--end", type=int, default=2020)
    ap.add_argument("--cells", type=int, default=40, help="top-N 0.75 deg cells by EDGAR 2022 CH4 emissions")
    ap.add_argument("--edgar", type=Path, default=EDGAR_DEFAULT)
    ap.add_argument("--out", type=Path, default=DEFAULT_CSV)
    a = ap.parse_args()
    df = fetch_copernicus(a.out, a.start, a.end, a.cells, a.edgar)
    print(f"wrote {a.out}: {len(df)} rows, {df.cell_id.nunique()} cells, "
          f"{df.date.min():%Y-%m}..{df.date.max():%Y-%m}")


if __name__ == "__main__":
    main()
