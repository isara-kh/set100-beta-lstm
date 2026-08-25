"""Diebold-Mariano test of equal predictive accuracy, LSTM vs each OLS benchmark.

Tests the monthly value-weighted loss differential
d_t = loss(LSTM) - loss(benchmark), so a negative mean (and DM statistic) means
the LSTM is more accurate. The Newey-West HAC standard error uses lag h - 1 = 11:
12-month-ahead errors overlap by construction, so d_t is MA(11) -- the lag is
structural, not tuned. The pre/post-COVID split cuts on the forecast origin
(pre <= 2019-12, post >= 2020-03, the two crisis-window origins dropped as a
buffer, following the Pisedtasalasai 2021 Thai crisis dates).
"""
import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.environ.get("TMPDIR", "/tmp"), "mpl"))

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from betaforecast.paths import OOS_FORECASTS as OOS, FIGURES, REPO

FIG = FIGURES / "cumulative_vw_differential.pdf"

H = 12                                   # forecast horizon (months)
LAG = H - 1                              # Newey-West truncation = MA(11), structural
COVID = "2020-03"                        # first post-COVID origin
PRE_END = "2019-12"                      # last pre-COVID origin
BENCHMARKS = {"OLS_1Y_D": "yhat_ols1yd", "OLS_5Y_M": "yhat_ols5ym"}


def monthly_differentials(df):
    """Return the value-weighted monthly loss-differential series per benchmark.

    d_t = loss(LSTM) - loss(benchmark), aggregated within each origin month using
    monthly market-cap shares (w = exp(ME) normalised per month).
    """
    df = df.copy()
    df["w"] = df.groupby("ym")["ME"].transform(lambda s: np.exp(s) / np.exp(s).sum())
    se_lstm = (df["y_true"] - df["yhat_lstm"]) ** 2

    vw = {}
    for name, col in BENCHMARKS.items():
        delta = se_lstm - (df["y_true"] - df[col]) ** 2
        vw[name] = (df["w"] * delta).groupby(df["ym"]).sum()
    return pd.DataFrame(vw).sort_index()


def dm_stat(d):
    """Diebold-Mariano statistic with Newey-West HAC standard error.

    Returns (dbar, DM, p_value, T) for one loss-differential series `d`;
    Bartlett kernel out to lag 11, referred to the standard normal.
    """
    d = np.asarray(d, float)
    T = d.size
    u = d - d.mean()
    gamma0 = (u @ u) / T
    lrv = gamma0
    for j in range(1, LAG + 1):
        gamma_j = (u[j:] @ u[:-j]) / T
        lrv += 2.0 * (1.0 - j / (LAG + 1)) * gamma_j      # Bartlett kernel
    se = np.sqrt(lrv / T)
    dm = d.mean() / se
    p = 2 * stats.norm.sf(abs(dm))
    return d.mean(), dm, p, T


def run_table(vw):
    """DM results for every pair x regime (value-weighted) as a tidy DataFrame."""
    regimes = {
        "full": lambda s: s,
        "pre":  lambda s: s[s.index <= PRE_END],
        "post": lambda s: s[s.index >= COVID],
    }
    rows = []
    for name in BENCHMARKS:
        for regime, sel in regimes.items():
            dbar, dm, p, T = dm_stat(sel(vw[name]))
            rows.append({"pair": f"LSTM-{name}", "regime": regime, "n": T,
                         "dbar": dbar, "DM": dm, "p_value": p})
    return pd.DataFrame(rows)


def plot_cumulative(vw):
    """Cumulative VW-MSE differential, benchmark minus LSTM (up = LSTM better)."""
    x = pd.PeriodIndex(vw.index, freq="M").to_timestamp()
    fig, ax = plt.subplots(figsize=(7, 4))
    styles = {"OLS_1Y_D": ("#2ca02c", "-"), "OLS_5Y_M": ("#d62728", "-")}
    for name in BENCHMARKS:
        c, ls = styles.get(name, ("black", "-"))
        ax.plot(x, (-vw[name]).cumsum(), color=c, ls=ls, lw=1.4,
                label=f"LSTM vs {name}")
    ax.axhline(0, color="k", lw=0.6)
    onset = pd.Timestamp("2020-03-01")
    ax.axvspan(onset, x.max(), color="grey", alpha=0.12, lw=0)
    ax.annotate("post-COVID", xy=(onset, ax.get_ylim()[1]),
                xytext=(4, -10), textcoords="offset points", fontsize=8, color="grey")
    ax.set_xlabel("Forecast origin")
    ax.set_ylabel("Cumulative VW-MSE differential\n(benchmark - LSTM; up = LSTM better)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG)
    plt.close(fig)


def plot_vw_mse_level(monthly):
    """Three-panel small multiples of each model's monthly VW-MSE (in percent):
    faint monthly series + bold centred six-month average, post-COVID shaded.
    """
    from matplotlib.lines import Line2D

    out = os.path.join(REPO, "figures", "vw_mse_over_time.pdf")
    m = monthly.copy() * 100.0
    m.index = pd.PeriodIndex(m.index, freq="M").to_timestamp()
    onset = pd.Timestamp("2020-03-01")
    series = [
        ("LSTM",     "#1f77b4"),  # blue
        ("OLS_1Y_D", "#2ca02c"),  # green
        ("OLS_5Y_M", "#d62728"),  # red
    ]
    fig, axes = plt.subplots(3, 1, figsize=(7, 5.5), sharex=True, sharey=True)
    for ax, (name, color) in zip(axes, series):
        ax.axvspan(onset, m.index.max(), color="grey", alpha=0.08, lw=0)
        ax.plot(m.index, m[name], color=color, lw=0.8, alpha=0.30)
        ax.plot(m.index, m[name].rolling(6, center=True, min_periods=1).mean(),
                color=color, lw=1.9)
        ax.set_title(name, color=color, fontsize=11, loc="left")
        for lbl in ax.get_yticklabels():
            lbl.set_fontweight("bold")
    axes[0].annotate("post-COVID", xy=(onset, axes[0].get_ylim()[1]),
                     xytext=(4, -10), textcoords="offset points",
                     fontsize=8, color="grey")
    grey_thin = Line2D([0], [0], color="0.4", lw=0.8, alpha=0.6)
    grey_bold = Line2D([0], [0], color="0.4", lw=1.9)
    axes[0].legend([grey_thin, grey_bold], ["Monthly", "6-month average"],
                   frameon=False, fontsize=9, loc="upper left")
    axes[1].set_ylabel("Value-weighted MSE (%)", fontweight="bold")
    axes[-1].set_xlabel("Forecast origin", fontweight="bold")
    for lbl in axes[-1].get_xticklabels():
        lbl.set_fontweight("bold")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return out


def report():
    df = pd.read_parquet(OOS)
    vw = monthly_differentials(df)
    res = run_table(vw)

    print(f"DM test: lag = {LAG} (MA(h-1), h = {H}), value-weighted, two-sided standard normal.")
    print("d_t = loss(LSTM) - loss(benchmark): dbar<0 => LSTM more accurate.\n")
    print(f"{'pair':16s}{'regime':7s}{'T':>4s}{'dbar':>10s}{'DM':>9s}{'p':>8s}")
    for _, r in res.iterrows():
        star = "*" if r.p_value < 0.05 else " "
        print(f"{r.pair:16s}{r.regime:7s}{r['n']:>4d}{r.dbar:>10.4f}"
              f"{r.DM:>9.2f}{r.p_value:>8.3f}{star}")
    print()

    plot_cumulative(vw)
    print(f"Cumulative VW differential -> {os.path.relpath(FIG, REPO)}")


if __name__ == "__main__":
    report()
