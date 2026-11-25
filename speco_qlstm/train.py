"""Train the QLSTM and the classical LSTM baseline, evaluate on the held-out final 20% of months, compare.

    python -m speco_qlstm.train --data csv --seeds 0 1 2
    python -m speco_qlstm.train --data synthetic --epochs 3 --seeds 0   # smoke test only
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

from .data import DEFAULT_CSV, FEATURES, Scaler, create_sequences, engineer, load, split
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
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    data, scaler, meta = prepare(a.data, a.csv, a.seq_len, a.horizon)
    print(f"data: {a.data} {json.dumps(meta)}")
    if a.data == "synthetic":
        print("NOTE: synthetic data — numbers below test the pipeline only and mean nothing about methane.")
    y_true = scaler.inverse_target(data["test"][1])

    runs = {"qlstm": [], "lstm": []}
    for seed in a.seeds:
        for name in ("lstm", "qlstm"):
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = build(name, len(FEATURES), a.hidden, a.horizon, a.qubits, a.layers)
            print(f"[seed {seed}] {name}: {sum(p.numel() for p in model.parameters())} parameters")
            hist = fit(model, data, a.epochs, a.lr, a.batch, a.patience)
            y_hat = scaler.inverse_target(predict(model, data["test"][0]))
            err = y_hat - y_true
            runs[name].append({"seed": seed, "mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean())),
                               "window_abs_err": np.abs(err).mean(1), "y_hat": y_hat, "hist": hist})
            r = runs[name][-1]
            print(f"[seed {seed}] {name}: test MAE {r['mae']:.3f} ppb  RMSE {r['rmse']:.3f} ppb")

    summary = {"meta": meta, "args": {k: str(v) for k, v in vars(a).items()}, "seeds": a.seeds}
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
    (a.out / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("args", "meta")}, indent=2))
    plots(runs, y_true, data["test"][2], a.out)


def plots(runs, y_true, cells, out: Path) -> None:
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
    fig.tight_layout(); fig.savefig(out / "loss_curves.png", dpi=150); plt.close(fig)

    q, c = runs["qlstm"][0]["y_hat"], runs["lstm"][0]["y_hat"]
    picks = [np.where(cells == u)[0][0] for u in np.unique(cells)[:4]]
    fig, axes = plt.subplots(1, len(picks), figsize=(4 * len(picks), 3.5), squeeze=False)
    for ax, i in zip(axes[0], picks):
        h = np.arange(1, y_true.shape[1] + 1)
        ax.plot(h, y_true[i], "k-o", label="observed"); ax.plot(h, q[i], "C0-s", label="QLSTM")
        ax.plot(h, c[i], "C1--^", label="LSTM")
        ax.set(title=f"cell {cells[i]}, first test window", xlabel="months ahead", ylabel="CH4 (ppb)")
    axes[0][0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "forecast_examples.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.boxplot([np.concatenate([r["window_abs_err"] for r in runs[n]]) for n in ("qlstm", "lstm")])
    ax.set_xticks([1, 2], ["QLSTM", "LSTM"])
    ax.set(ylabel="mean absolute error per window (ppb)", title="Test error, all seeds")
    fig.tight_layout(); fig.savefig(out / "error_boxplot.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
