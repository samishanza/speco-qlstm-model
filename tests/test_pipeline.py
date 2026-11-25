import numpy as np
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
