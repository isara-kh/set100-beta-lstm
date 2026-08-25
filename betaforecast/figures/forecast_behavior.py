"""Forecast-behaviour figures from oos_forecasts.parquet: pooled densities of
realized beta vs the three forecasts (forecast_distribution.pdf), their monthly
cross-sectional averages over 2015-2024 (forecast_timeseries.pdf), and the
forecast-vs-realized correlation table.
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from betaforecast.evaluation import dm_test as DM       # regime boundaries

from betaforecast.paths import (OOS_FORECASTS as OOS, FIGURES as FIGDIR,
                                DATA_PROCESSED, REPO)

CORR = DATA_PROCESSED / "forecast_correlations.parquet"

FORECASTS = [
    ("LSTM",     "yhat_lstm"),
    ("OLS_1Y_D", "yhat_ols1yd"),
    ("OLS_5Y_M", "yhat_ols5ym"),
]

# series label -> (column, colour, linestyle)
SERIES = [
    ("Realized beta", "y_true",      "black",     "-"),
    ("LSTM",          "yhat_lstm",   "#1f77b4",   "-"),   # blue
    ("OLS_1Y_D",      "yhat_ols1yd", "#2ca02c",   "-"),   # green
    ("OLS_5Y_M",      "yhat_ols5ym", "#d62728",   "-"),   # red
]


def plot_distribution(df):
    """Single overlay: realized beta (grey reference) and the three forecasts."""
    grid = np.linspace(-0.2, 2.6, 400)
    rt = df["y_true"].dropna().values
    rt_kde = gaussian_kde(rt)(grid)

    forecasts = [
        ("LSTM",       "yhat_lstm",   "#1f77b4"),  # blue
        ("OLS_1Y_D",   "yhat_ols1yd", "#2ca02c"),  # green
        ("OLS_5Y_M",   "yhat_ols5ym", "#d62728"),  # red
    ]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    # realized beta as a filled grey reference distribution
    ax.fill_between(grid, rt_kde, color="0.75", alpha=0.6, lw=0, label="Realized beta")
    ax.plot(grid, rt_kde, color="0.5", lw=1.2)
    ax.axvline(rt.mean(), color="0.5", ls=":", lw=0.9)
    # the three forecasts as clean lines
    for label, col, color in forecasts:
        x = df[col].dropna().values
        ax.plot(grid, gaussian_kde(x)(grid), color=color, lw=1.8, label=label)
        ax.axvline(x.mean(), color=color, ls=":", lw=0.9, alpha=0.7)
    ax.set_xlabel(r"Beta ($\beta$)", fontweight="bold")
    ax.set_ylabel("Density", fontweight="bold")
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
    ax.set_xlim(-0.2, 2.6)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "forecast_distribution.pdf")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return out


def plot_timeseries(df):
    """Small multiples over time: realized beta vs one forecast per panel."""
    cols = [c for _, c, _, _ in SERIES]
    g = df.groupby("ym")[cols].mean().sort_index()
    x = pd.PeriodIndex(g.index, freq="M").to_timestamp()
    onset = pd.Timestamp("2020-03-01")

    forecasts = [
        ("LSTM",     "yhat_lstm",   "#1f77b4"),  # blue
        ("OLS_1Y_D", "yhat_ols1yd", "#2ca02c"),  # green
        ("OLS_5Y_M", "yhat_ols5ym", "#d62728"),  # red
    ]
    fig, axes = plt.subplots(3, 1, figsize=(7, 5.5), sharex=True, sharey=True)
    realized_handle = None
    for ax, (label, col, color) in zip(axes, forecasts):
        ax.axvspan(onset, x.max(), color="grey", alpha=0.08, lw=0)
        realized_handle, = ax.plot(x, g["y_true"], color="black", lw=1.3)
        ax.plot(x, g[col], color=color, lw=1.3)
        ax.set_title(label, color=color, fontsize=11, loc="left")
        for lbl in ax.get_yticklabels():
            lbl.set_fontweight("bold")
    axes[0].annotate("pre-COVID", xy=(onset, axes[0].get_ylim()[1]),
                     xytext=(-4, -10), textcoords="offset points",
                     fontsize=8, color="grey", ha="right")
    axes[0].annotate("post-COVID", xy=(onset, axes[0].get_ylim()[1]),
                     xytext=(4, -10), textcoords="offset points",
                     fontsize=8, color="grey")
    axes[0].legend([realized_handle], ["Realized beta"],
                   frameon=False, fontsize=9, loc="upper left")
    axes[1].set_ylabel("Cross-sectional average beta", fontweight="bold")
    axes[-1].set_xlabel("Forecast origin", fontweight="bold")
    for lbl in axes[-1].get_xticklabels():
        lbl.set_fontweight("bold")
    fig.tight_layout()
    out = os.path.join(FIGDIR, "forecast_timeseries.pdf")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return out


def plot_timeseries_rebased(df):
    """Single panel: cross-sectional average of realized beta and the three
    forecasts, each indexed to 100 at the first OOS month so only relative
    movements remain."""
    cols = [c for _, c, _, _ in SERIES]
    g = df.groupby("ym")[cols].mean().sort_index()
    g = g / g.iloc[0] * 100.0
    x = pd.PeriodIndex(g.index, freq="M").to_timestamp()
    onset = pd.Timestamp("2020-03-01")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.axvspan(onset, x.max(), color="grey", alpha=0.08, lw=0)
    ax.axhline(100, color="k", lw=0.6, ls=":")
    for label, col, color, ls in SERIES:
        ax.plot(x, g[col], color=color, lw=1.5, ls=ls, label=label)
    ax.annotate("pre-COVID", xy=(onset, ax.get_ylim()[1]), xytext=(-4, -10),
                textcoords="offset points", fontsize=8, color="grey", ha="right")
    ax.annotate("post-COVID", xy=(onset, ax.get_ylim()[1]), xytext=(4, -10),
                textcoords="offset points", fontsize=8, color="grey")
    ax.set_xlabel("Forecast origin", fontweight="bold")
    ax.set_ylabel("Average beta (Jan 2015 = 100)", fontweight="bold")
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "forecast_timeseries_rebased.pdf")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return out


def _corrs(sub):
    """Correlation of each forecast with the realized beta over one subsample, plus
    the agreement between the two short-window forecasts. Pearson (r) and Spearman
    (rho); the latter is the non-parametric cross-check robust to the fat tails."""
    row = {"n": len(sub)}
    for name, col in FORECASTS:
        row[f"{name}_r"] = sub["y_true"].corr(sub[col])
        row[f"{name}_rho"] = sub["y_true"].corr(sub[col], method="spearman")
    row["LSTM_OLS1YD_r"] = sub["yhat_lstm"].corr(sub["yhat_ols1yd"])
    return row


def correlation_table(df):
    """Forecast correlations with realized beta, pooled and split pre/post-COVID
    on the forecast origin (same regime boundaries as the DM test).
    """
    ym = df["ym"].astype(str)
    blocks = {
        "pooled": df,
        "pre": df[ym <= DM.PRE_END],
        "post": df[ym >= DM.COVID],
    }
    return pd.DataFrame({k: _corrs(v) for k, v in blocks.items()}).T


def report():
    df = pd.read_parquet(OOS)
    d = plot_distribution(df)
    t = plot_timeseries(df)
    tr = plot_timeseries_rebased(df)
    print("Figures ->", os.path.relpath(d, REPO), "|", os.path.relpath(t, REPO),
          "|", os.path.relpath(tr, REPO))

    corr = correlation_table(df)
    corr.to_parquet(CORR)
    print(f"\nForecast correlations with realized beta -> {os.path.relpath(CORR, REPO)}")
    print(corr.round(4).to_string())


if __name__ == "__main__":
    report()
