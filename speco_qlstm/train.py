"""Train the QLSTM and the classical LSTM baseline, evaluate on the held-out final 20% of months, compare.

    python -m speco_qlstm.train --data csv --seeds 0 1 2
    python -m speco_qlstm.train --data csv --target seasonal-diff        # predict y(t+k) - y(t+k-12)
    python -m speco_qlstm.train --data synthetic --epochs 3 --seeds 0   # smoke test only

Targets:
- level (default): the model predicts min-max-scaled CH4 for the next `horizon` months.
- seasonal-diff: the model predicts, for each horizon step k, the residual y(t+k) - y(t+k-12) against the same month
  one year earlier, standardised with a scaler fit on training windows only. The ppb forecast is the predicted
  residual plus y(t+k-12), which is always inside the input window (asserted), so no future value is used.
Metrics are always in ppb on the same test windows, next to persistence and seasonal-naive baselines.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from torch import nn

from .data import DEFAULT_CSV, FEATURES, TARGET, Scaler, create_sequences, engineer, load, split
from .models import build

ROOT = Path(__file__).resolve().parents[1]


def prepare(source: str, csv: Path, seq_len: int, horizon: int, val_frac: float = 0.15):
    df = engineer(load(source, csv))
    train_df, test_df = split(df, 0.8)
    scaler = Scaler.fit(train_df)
    train_s, test_s = scaler.transform(train_df), scaler.transform(test_df)
    # validation = last val_frac of training months (still strictly before the test period)
    tr_core, val = split(train_s, 1 - val_frac)
    data = {
        "train": create_sequences(tr_core, seq_len, horizon),
        "val": create_sequences(val, seq_len, horizon, context=tr_core),
        "test": create_sequences(test_s, seq_len, horizon, context=train_s),
    }
    meta = {"rows": len(df), "cells": int(df.cell_id.nunique()),
            "train_months": f"{train_df.date.min():%Y-%m}..{train_df.date.max():%Y-%m}",
            "test_months": f"{test_df.date.min():%Y-%m}..{test_df.date.max():%Y-%m}",
            **{f"n_{k}": int(len(v[0])) for k, v in data.items()}}
    return data, scaler, meta


CH4_COL = FEATURES.index(TARGET)


def seasonal_base(x: np.ndarray, horizon: int, scaler: Scaler) -> np.ndarray:
    """y(t+k-12) in ppb for k = 0..horizon-1, read from the input window x (n, seq_len, features).

    Window month j (0-based) is t - seq_len + j, so t+k-12 is window index seq_len - 12 + k. It must lie inside the
    window: seq_len >= 12 (not before the window) and horizon <= 12 (not after the last input month).
    """
    seq_len = x.shape[1]
    idx = seq_len - 12 + np.arange(horizon)
    assert idx.min() >= 0 and idx.max() <= seq_len - 1, "seasonal base would fall outside the input window"
    return scaler.inverse_target(x[:, idx, CH4_COL])


def persistence(x: np.ndarray, horizon: int, scaler: Scaler) -> np.ndarray:
    """Last observed month repeated for every horizon step (ppb)."""
    return np.repeat(scaler.inverse_target(x[:, -1, CH4_COL])[:, None], horizon, axis=1)


class ResidualTarget:
    """Seasonal-difference target: r = y - y_lastyear (ppb), standardised with training-window statistics."""

    def __init__(self, data, scaler: Scaler, horizon: int):
        self.scaler, self.horizon = scaler, horizon
        r = self.residual(*data["train"][:2])
        self.mu, self.sd = float(r.mean()), float(r.std()) or 1.0

    def residual(self, x, y_scaled):
        return self.scaler.inverse_target(y_scaled) - seasonal_base(x, self.horizon, self.scaler)

    def encode(self, x, y_scaled) -> np.ndarray:
        return ((self.residual(x, y_scaled) - self.mu) / self.sd).astype(np.float32)

    def to_ppb(self, x, pred) -> np.ndarray:
        return pred * self.sd + self.mu + seasonal_base(x, self.horizon, self.scaler)


def fit(model: nn.Module, data, epochs: int, lr: float, batch: int, patience: int, log=print):
    xtr, ytr = map(torch.from_numpy, data["train"][:2])
    xva, yva = map(torch.from_numpy, data["val"][:2])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best, best_state, bad = float("inf"), None, 0
    hist = {"train": [], "val": [], "sec_per_epoch": []}
    for ep in range(epochs):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(xtr))
        tot = 0.0
        for k in range(0, len(xtr), batch):
            idx = perm[k:k + batch]
            opt.zero_grad()
            loss = loss_fn(model(xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(xva), yva).item() if len(xva) else float("nan")
        hist["train"].append(tot / len(xtr))
        hist["val"].append(vl)
        hist["sec_per_epoch"].append(time.time() - t0)
        log(f"  epoch {ep + 1:3d}  train {hist['train'][-1]:.5f}  val {vl:.5f}  ({hist['sec_per_epoch'][-1]:.1f}s)")
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return hist


def predict(model, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(x)).numpy()


def _rel(v):
    """Record paths relative to the repo so results files carry no local home path."""
    if isinstance(v, Path):
        try:
            return str(v.resolve().relative_to(ROOT.resolve()))
        except ValueError:
            return v.name
    return str(v)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", choices=["csv", "synthetic"], default="csv")
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=5)
    ap.add_argument("--qubits", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=12)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--target", choices=["level", "seasonal-diff"], default="level")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    ap.add_argument("--tag", default="", help="suffix for metrics/plot file names, e.g. level_e40")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    data, scaler, meta = prepare(a.data, a.csv, a.seq_len, a.horizon)
    print(f"data: {a.data} {json.dumps(meta)}")
    if a.data == "synthetic":
        print("NOTE: synthetic data — numbers below test the pipeline only and mean nothing about methane.")
    x_test = data["test"][0]
    y_true = scaler.inverse_target(data["test"][1])
    if a.target == "seasonal-diff":
        rt = ResidualTarget(data, scaler, a.horizon)
        data = {k: (x, rt.encode(x, y), c) for k, (x, y, c) in data.items()}
        to_ppb = rt.to_ppb
        meta["residual_scaler"] = {"mean_ppb": rt.mu, "sd_ppb": rt.sd}
    else:
        to_ppb = lambda x, pred: scaler.inverse_target(pred)  # noqa: E731
    baselines = {}
    for bname, f in (("persistence", persistence(x_test, a.horizon, scaler)),
                     ("seasonal_naive", seasonal_base(x_test, a.horizon, scaler))):
        err = f - y_true
        baselines[bname] = {"mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean()))}
        print(f"baseline {bname}: test MAE {baselines[bname]['mae']:.3f} ppb  RMSE {baselines[bname]['rmse']:.3f} ppb")

    runs = {"qlstm": [], "lstm": []}
    for seed in a.seeds:
        for name in ("lstm", "qlstm"):
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = build(name, len(FEATURES), a.hidden, a.horizon, a.qubits, a.layers)
            print(f"[seed {seed}] {name}: {sum(p.numel() for p in model.parameters())} parameters")
            hist = fit(model, data, a.epochs, a.lr, a.batch, a.patience)
            y_hat = to_ppb(x_test, predict(model, x_test))
            err = y_hat - y_true
            runs[name].append({"seed": seed, "mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean())),
                               "window_abs_err": np.abs(err).mean(1), "y_hat": y_hat, "hist": hist})
            r = runs[name][-1]
            print(f"[seed {seed}] {name}: test MAE {r['mae']:.3f} ppb  RMSE {r['rmse']:.3f} ppb")

    summary = {"meta": meta, "args": {k: _rel(v) for k, v in vars(a).items()}, "seeds": a.seeds,
               "target": a.target, "baselines": baselines}
    for name in runs:
        summary[name] = {"mae": [r["mae"] for r in runs[name]], "rmse": [r["rmse"] for r in runs[name]],
                         "mae_mean": float(np.mean([r["mae"] for r in runs[name]])),
                         "rmse_mean": float(np.mean([r["rmse"] for r in runs[name]])),
                         "sec_per_epoch_mean": float(np.mean([np.mean(r["hist"]["sec_per_epoch"]) for r in runs[name]]))}
    q, c = summary["qlstm"], summary["lstm"]
    summary["rmse_change_pct_qlstm_vs_lstm"] = 100 * (q["rmse_mean"] - c["rmse_mean"]) / c["rmse_mean"]
    summary["mae_change_pct_qlstm_vs_lstm"] = 100 * (q["mae_mean"] - c["mae_mean"]) / c["mae_mean"]
    if len(a.seeds) > 1:
        for m in ("mae", "rmse"):
            t = stats.ttest_rel(q[m], c[m])
            summary[f"ttest_seeds_{m}"] = {"t": float(t.statistic), "p": float(t.pvalue)}
    qa = np.mean([r["window_abs_err"] for r in runs["qlstm"]], 0)
    ca = np.mean([r["window_abs_err"] for r in runs["lstm"]], 0)
    t = stats.ttest_rel(qa, ca)
    summary["ttest_windows_abs_err"] = {"t": float(t.statistic), "p": float(t.pvalue), "n": int(len(qa))}
    sfx = f"_{a.tag}" if a.tag else ""
    for name in runs:
        s_ = summary[name]
        for bname, b in baselines.items():
            s_[f"mae_vs_{bname}_pct"] = 100 * (s_["mae_mean"] - b["mae"]) / b["mae"]
            s_[f"rmse_vs_{bname}_pct"] = 100 * (s_["rmse_mean"] - b["rmse"]) / b["rmse"]
    (a.out / f"metrics{sfx}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("args", "meta")}, indent=2))
    plots(runs, y_true, data["test"][2], a.out, sfx)


def plots(runs, y_true, cells, out: Path, sfx: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, style in (("qlstm", "-"), ("lstm", "--")):
        for r in runs[name]:
            ax.plot(r["hist"]["train"], style, color="C0", alpha=.6, label=f"{name} train (seed {r['seed']})")
            ax.plot(r["hist"]["val"], style, color="C1", alpha=.6, label=f"{name} val (seed {r['seed']})")
    ax.set(xlabel="epoch", ylabel="MSE (scaled)", title="Training and validation loss", yscale="log")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(out / f"loss_curves{sfx}.png", dpi=150); plt.close(fig)

    q, c = runs["qlstm"][0]["y_hat"], runs["lstm"][0]["y_hat"]
    picks = [np.where(cells == u)[0][0] for u in np.unique(cells)[:4]]
    fig, axes = plt.subplots(1, len(picks), figsize=(4 * len(picks), 3.5), squeeze=False)
    for ax, i in zip(axes[0], picks):
        h = np.arange(1, y_true.shape[1] + 1)
        ax.plot(h, y_true[i], "k-o", label="observed"); ax.plot(h, q[i], "C0-s", label="QLSTM")
        ax.plot(h, c[i], "C1--^", label="LSTM")
        ax.set(title=f"cell {cells[i]}, first test window", xlabel="months ahead", ylabel="CH4 (ppb)")
    axes[0][0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / f"forecast_examples{sfx}.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.boxplot([np.concatenate([r["window_abs_err"] for r in runs[n]]) for n in ("qlstm", "lstm")])
    ax.set_xticks([1, 2], ["QLSTM", "LSTM"])
    ax.set(ylabel="mean absolute error per window (ppb)", title="Test error, all seeds")
    fig.tight_layout(); fig.savefig(out / f"error_boxplot{sfx}.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
