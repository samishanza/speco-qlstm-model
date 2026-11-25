import numpy as np
import pytest
import torch

from speco_qlstm.data import FEATURES, Scaler, create_sequences, engineer, split, synthetic_dataset
from speco_qlstm.models import build
from speco_qlstm.train import fit, prepare


def test_split_is_chronological_and_scaler_uses_train_only():
    df = engineer(synthetic_dataset())
    tr, te = split(df)
    assert tr.date.max() < te.date.min()
    s = Scaler.fit(tr)
    assert s.hi["ch4"] == tr.ch4.max()  # not influenced by later (higher, trending) test values
    assert te.ch4.max() > tr.ch4.max()


def test_rolling_features_use_only_the_past():
    df = engineer(synthetic_dataset(n_cells=1))
    raw = synthetic_dataset(n_cells=1).set_index("date")["ch4"]
    row = df.iloc[10]
    past = raw[raw.index < row.date].tail(6)
    assert np.isclose(row.ch4_roll_mean6, past.mean())
    assert np.isclose(row.ch4_lag1, past.iloc[-1])


def test_sequences_shapes_and_alignment():
    df = engineer(synthetic_dataset(n_cells=2, n_months=40))
    X, y, cells = create_sequences(df, seq_len=12, horizon=6)
    per_cell = len(df) // 2 - 12 - 6 + 1
    assert X.shape == (2 * per_cell, 12, len(FEATURES)) and y.shape == (2 * per_cell, 6)
    d0 = df[df.cell_id == 0].sort_values("date")
    assert np.allclose(y[0], d0.ch4.to_numpy()[12:18])


def test_test_targets_never_come_from_training_rows():
    df = engineer(synthetic_dataset(n_cells=1))
    tr, te = split(df)
    X, y, _ = create_sequences(te, 12, 6, context=tr)
    assert len(y) == len(te) - 6 + 1
    assert np.allclose(y[0], te.sort_values("date").ch4.to_numpy()[:6])


def test_models_forward_shapes():
    x = torch.randn(3, 12, len(FEATURES))
    for name in ("qlstm", "lstm"):
        assert build(name, len(FEATURES), 5, 6).forward(x).shape == (3, 6)


def test_one_epoch_smoke_both_models():
    data, _, _ = prepare("synthetic", None, 12, 6)
    for name in ("qlstm", "lstm"):
        torch.manual_seed(0)
        h = fit(build(name, len(FEATURES), 5, 6), data, epochs=1, lr=0.01, batch=64, patience=1, log=lambda *_: None)
        assert np.isfinite(h["train"][0])


# ---- NOAA CarbonTracker-CH4 fetch helpers (no network) ----

import pandas as pd
import xarray as xr

from speco_qlstm.data import (CT_LAT_IDX, CT_LATS, CT_LON_IDX, CT_LONS, CT_SHAPE, block_mean, ctch4_byte_ranges,
                              monthly_mean_of_days, parse_multipart_ranges, pick_cells, tidy_rows)


def test_ct_grid_covers_conus_box():
    assert CT_LATS[CT_LAT_IDX].tolist() == list(range(25, 50, 2))
    assert CT_LONS[CT_LON_IDX][0] == -124.5 and CT_LONS[CT_LON_IDX][-1] == -67.5 and len(CT_LON_IDX) == 20


def test_byte_ranges_match_c_order_offsets():
    # simulate the file: ch4 values = their own flat C-order index, stored as <f8 after a fixed offset
    from speco_qlstm import data as D
    flat = np.arange(np.prod(CT_SHAPE), dtype="<f8")
    blob = b"\0" * D.CT_CH4_OFFSET + flat.tobytes()
    ref = flat.reshape(CT_SHAPE)[:, 0, CT_LAT_IDX[0]:CT_LAT_IDX[-1] + 1, :]
    for t, (a, b) in enumerate(ctch4_byte_ranges()):
        got = np.frombuffer(blob[a:b + 1], "<f8").reshape(len(CT_LAT_IDX), CT_SHAPE[3])
        assert np.array_equal(got, ref[t])


def test_parse_multipart_ranges():
    body = (b"\r\n--XYZ\r\nContent-Type: application/x-netcdf\r\nContent-Range: bytes 10-13/100\r\n\r\nabcd"
            b"\r\n--XYZ\r\nContent-Type: application/x-netcdf\r\nContent-Range: bytes 50-51/100\r\n\r\n\r\n"
            b"\r\n--XYZ--\r\n")
    parts = parse_multipart_ranges(body, "multipart/byteranges; boundary=XYZ")
    assert parts == {10: (100, b"abcd"), 50: (100, b"\r\n")}


def test_monthly_mean_of_days_and_missing_day():
    days = {d: np.full((8, 2, 2), float(d.day)) for d in pd.date_range("2010-02-01", "2010-02-28")}
    t, m = monthly_mean_of_days(days)
    assert t[0] == pd.Timestamp("2010-02-01") and np.allclose(m[0], 14.5)
    days.pop(pd.Timestamp("2010-02-10"))
    with pytest.raises(ValueError):
        monthly_mean_of_days(days)


def test_block_mean_half_open_cells():
    lat = np.arange(24, 28.01, 0.75)  # 24, 24.75, ..., 27.75
    lon = np.arange(-126, -120, 0.75)
    v = np.add.outer(lat, np.zeros_like(lon))[None]
    da = xr.DataArray(v, dims=("time", "lat", "lon"), coords={"time": [0], "lat": lat, "lon": lon})
    out = block_mean(da, np.array([25.0, 27.0]), np.array([-124.5, -121.5]), 2.0, 3.0)
    assert np.allclose(out.values[0, 0], np.mean([24, 24.75, 25.5]))   # [24, 26)
    assert np.allclose(out.values[0, 1], np.mean([26.25, 27, 27.75]))  # [26, 28)


def test_pick_cells_and_tidy_rows():
    emis = np.array([[5.0, 1.0], [9.0, 7.0]])
    valid = np.array([[True, True], [False, True]])
    assert pick_cells(emis, valid, 2) == [(1, 1), (0, 0)]
    n = 3
    ones = np.ones((n, 2, 2))
    df = tidy_rows([(1, 1)], np.array([25.0, 27.0]), np.array([-100.5, -97.5]),
                   pd.date_range("2010-01-15", periods=n, freq="MS"), 1900 * ones, 3 * ones, 290 * ones,
                   1e5 * ones, emis)
    assert list(df.columns) == ["cell_id", "lat", "lon", "date", "ch4", "wind_speed", "temperature", "pressure",
                                "edgar_emission"]
    r = df.iloc[0]
    assert (r.lat, r.lon, r.edgar_emission) == (27.0, -97.5, 7.0)
    assert np.isclose(r.temperature, 16.85) and np.isclose(r.pressure, 1000.0) and r.date.day == 1


def test_fetch_open_methane_end_to_end_offline(tmp_path, monkeypatch):
    """Glue code: CT-CH4 monthly file + ERA5-like file (CDS netCDF naming) + EDGAR -> tidy CSV on the CT grid."""
    from speco_qlstm import data as D
    times = pd.date_range("2010-01-01", periods=3, freq="MS")
    lats, lons = CT_LATS[CT_LAT_IDX], CT_LONS[CT_LON_IDX]
    ch4 = xr.DataArray(1900 + np.random.default_rng(0).random((3, len(lats), len(lons))),
                       dims=("time", "lat", "lon"), coords={"time": times, "lat": lats, "lon": lons}, name="ch4")
    ct_file = tmp_path / "ct.nc"
    ch4.to_dataset().to_netcdf(ct_file)
    elat, elon = np.arange(50, 23.9, -0.75), np.arange(-125, -65.9, 0.75)
    shp = (3, len(elat), len(elon))
    met = xr.Dataset({k: (("valid_time", "latitude", "longitude"), np.full(shp, v))
                      for k, v in (("u10", 3.0), ("v10", 4.0), ("t2m", 283.15), ("sp", 95000.0))},
                     coords={"valid_time": times + pd.Timedelta("6h"), "latitude": elat, "longitude": elon})
    era_file = tmp_path / "era.nc"
    met.to_netcdf(era_file)
    glat, glon = np.arange(-89.95, 90, 0.1), np.arange(-179.95, 180, 0.1)
    em = np.zeros((len(glat), len(glon)), np.float32)
    em[np.abs(glat - 31.0).argmin(), np.abs(glon - (-103.5)).argmin()] = 50.0   # Permian-ish cell wins
    edgar = tmp_path / "edgar.nc"
    xr.Dataset({"emissions": (("lat", "lon"), em)}, coords={"lat": glat, "lon": glon}).to_netcdf(edgar)
    monkeypatch.setattr(D, "download_ctch4_monthly", lambda *a, **k: ct_file)
    monkeypatch.setattr(D, "download_era5", lambda *a, **k: era_file)
    out = D.fetch_open_methane(tmp_path / "out.csv", 2010, 2010, n_cells=3, edgar_path=edgar)
    assert out.cell_id.nunique() == 3 and len(out) == 9
    top = out[out.cell_id == 0].iloc[0]
    assert (top.lat, top.lon) == (31.0, -103.5) and np.isclose(top.edgar_emission, 50.0)
    assert np.allclose(out.wind_speed, 5.0) and np.allclose(out.temperature, 10.0) and np.allclose(out.pressure, 950.0)
    assert np.isclose(top.ch4, ch4.sel(lat=31.0, lon=-103.5).values[0])
