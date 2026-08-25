"""Forecast accuracy by size tercile.

Each month the out-of-sample stocks are sorted into small/mid/large by market
cap; the VW-MSE and a Diebold-Mariano test (LSTM vs each benchmark) are computed
within each group, showing where the accuracy gap actually lies.
"""

import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from betaforecast.evaluation import dm_test  # Diebold-Mariano test, Newey-West lag 11
from betaforecast.paths import OOS_FORECASTS as OOS, FIGURES, REPO

FIG = FIGURES / "size_subsample_mse.pdf"

MODELS = [("LSTM", "yhat_lstm", "#1f77b4"),
          ("OLS_1Y_D", "yhat_ols1yd", "#2ca02c"),
          ("OLS_5Y_M", "yhat_ols5ym", "#d62728")]
GROUPS = ["Small", "Mid", "Large"]


def assign_terciles(df):
    """Within each origin month, split stocks into 3 size groups by ME."""
    df = df.copy()
    df["size_grp"] = (
        df.groupby("ym")["ME"]
        .transform(lambda s: pd.qcut(s.rank(method="first"), 3,
                                     labels=GROUPS))
    )
    return df


def _mse(sub, col, weighted):
    """MSE for one model within a sub-frame: pooled equal-weighted
    (weighted=False) or the study's monthly value-weighted MSE (weighted=True).
    """
    se = (sub["y_true"] - sub[col]) ** 2
    if not weighted:
        return float(se.mean())
    w = sub.groupby("ym")["cap"].transform(lambda s: s / s.sum())
    return float((w * se).groupby(sub["ym"]).sum().mean())


def group_mse(df, weighted):
    """MSE per (size group, model) under the chosen weighting."""
    rows = []
    for g in GROUPS:
        sub = df[df["size_grp"] == g]
        row = {"group": g, "avg_stocks": sub.groupby("ym").size().mean()}
        for name, col, _ in MODELS:
            row[name] = _mse(sub, col, weighted)
        rows.append(row)
    return pd.DataFrame(rows).set_index("group")


def group_dm(df, weighted):
    """Diebold-Mariano test of LSTM vs each benchmark within each size group,
    on the monthly (value- or equal-weighted) loss differential; negative = LSTM
    more accurate.
    """
    rows = []
    for g in GROUPS:
        sub = df[df["size_grp"] == g]
        se_lstm = (sub["y_true"] - sub["yhat_lstm"]) ** 2
        w = sub.groupby("ym")["cap"].transform(lambda s: s / s.sum())
        row = {"group": g}
        for name, col, _ in MODELS[1:]:
            delta = se_lstm - (sub["y_true"] - sub[col]) ** 2
            if weighted:
                d = (w * delta).groupby(sub["ym"]).sum().sort_index()
            else:
                d = delta.groupby(sub["ym"]).mean().sort_index()
            dbar, dm, p, T = dm_test.dm_stat(d)
            row[f"DM_{name}"] = dm
            row[f"p_{name}"] = p
        rows.append(row)
    return pd.DataFrame(rows).set_index("group")


def plot_group_mse(vw):
    """Grouped bars of the value-weighted MSE by size tercile."""
    fig, ax = plt.subplots(figsize=(6, 4.2))
    x = np.arange(len(GROUPS))
    wbar = 0.26
    for i, (name, _, color) in enumerate(MODELS):
        ax.bar(x + (i - 1) * wbar, vw[name].values, wbar,
               label=name, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(GROUPS)
    ax.set_xlabel("Size group (small to large)")
    ax.set_ylabel("Value-weighted MSE")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG)
    plt.close(fig)
    return FIG


def report():
    df = assign_terciles(pd.read_parquet(OOS))
    df["cap"] = np.exp(df["ME"])
    vw = group_mse(df, True)
    dm_vw = group_dm(df, True)

    print("Value-weighted MSE by size tercile (sorted each month by ME):\n")
    print(vw.round(4).to_string())
    print("\nDM (LSTM vs benchmark) per group, neg => LSTM more accurate:")
    print("\nValue-weighted differential:\n", dm_vw.round(3).to_string())

    out = plot_group_mse(vw)
    print(f"\nFigure -> {os.path.relpath(out, REPO)}")


if __name__ == "__main__":
    report()
