"""Replicas of Drobetz et al. (2025) Figures 1 and 5 with our SET100 forecasts.

Figure 1: relative forecast error 1 - MSE(LSTM)/MSE(benchmark) per origin month
(positive = LSTM better), monthly and calendar-year averaged, post-COVID shaded.
Figure 5: value-weighted zero-out importance, clipped at zero and normalised to
sum to 1, by predictor group and individually. Reads parquets only -- run after
evaluation.evaluate and importance.zero_out.
"""
import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.environ.get("TMPDIR", "/tmp"), "mpl"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from betaforecast.paths import (MONTHLY_VW_MSE as MSE, PREDICTOR_IMPORTANCE as IMP,
                                FIGURES, REPO)

FIG1 = FIGURES / "drobetz_fig1_relative_errors.pdf"
FIG5 = FIGURES / "drobetz_fig5_importance.pdf"

# Post-COVID information regime begins at the Mar. 2020 forecast origin.
COVID = pd.Timestamp("2020-03-01")

# Drobetz predictor groups mapped onto our ten predictors.
GROUPS = [
    ("Predictors Based on Sample Estimates of Beta", ["OLS_3M_D", "OLS_1Y_D", "OLS_5Y_M"], "#33a02c"),
    ("Technical Indicators",                          ["TO", "MOM", "IVOL", "ILLIQ"],       "#f5d800"),
    ("Predictors Based on Accounting Information",     ["ME", "DIVPAY"],                     "#b2df8a"),
    ("Macroeconomic Indicators",                      ["DFY"],                              "#d62728"),
]


# --------------------------------------------------------------------------- Figure 1
def relative_errors():
    """Monthly and calendar-year relative forecast errors vs each benchmark."""
    m = pd.read_parquet(MSE).copy()
    m.index = pd.PeriodIndex(m.index, freq="M").to_timestamp()    # ym -> month start
    out = {}
    for bench in ("OLS_1Y_D", "OLS_5Y_M"):
        monthly = 1.0 - m["LSTM"] / m[bench]
        # average MSE within each calendar year, then form the relative error
        yr = m.groupby(m.index.year)[["LSTM", bench]].mean()
        yearly = 1.0 - yr["LSTM"] / yr[bench]
        out[bench] = (monthly, yearly)
    return out


def plot_fig1():
    err = relative_errors()
    panels = [("OLS_1Y_D", "Graph A. LSTM vs. OLS_1Y_D"),
              ("OLS_5Y_M", "Graph B. LSTM vs. OLS_5Y_M")]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.0), sharex=True)

    for ax, (bench, title) in zip(axes, panels):
        monthly, yearly = err[bench]
        end = monthly.index.max() + pd.offsets.MonthEnd(1)

        ax.axvspan(COVID, end, color="0.85", zorder=0)           # post-COVID band
        ax.axhline(0, color="0.4", lw=0.8, zorder=1)

        # monthly relative error, with re-estimation-date markers (Drobetz circles)
        ax.plot(monthly.index, monthly.values, color="#1f77b4", lw=1.0,
                marker="o", mfc="none", mec="#e08214", ms=4, mew=0.8,
                label="MSE_MONTHLY_PD", zorder=3)

        # calendar-year average, drawn as a step held across each year
        xs, ys = [], []
        for y, v in yearly.items():
            xs += [pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-31")]
            ys += [v, v]
        ax.plot(xs, ys, color="k", lw=2.0, label="MSE_TEST_PD", zorder=4)

        ax.set_title(title, fontsize=11, style="italic")
        ax.set_ylabel(f"Percentage Difference\n(LSTM vs. {bench})")
        ax.margins(x=0.01)

    axes[0].legend(loc="upper right", fontsize=8, frameon=False)
    axes[1].set_xlabel("Re-Estimation Date")
    fig.suptitle("Relative Forecast Errors over Time", fontsize=13, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG1)
    plt.close(fig)


# --------------------------------------------------------------------------- Figure 5
def relative_importance():
    """Clip absolute VI at zero and normalise to sum to 1 (relative contribution)."""
    imp = pd.read_parquet(IMP)["VI_zeroout"]
    rel = imp.clip(lower=0)
    rel = rel / rel.sum()
    return imp, rel


def plot_fig5():
    imp, rel = relative_importance()

    fig, (axA, axB) = plt.subplots(2, 1, figsize=(7.2, 6.2),
                                   gridspec_kw={"height_ratios": [1, 3]})

    # Graph A: predictor groups as one stacked horizontal bar summing to 1
    left = 0.0
    handles = []
    for name, members, color in GROUPS:
        share = rel.reindex(members).fillna(0).sum()
        axA.barh(0, share, left=left, color=color, edgecolor="k", lw=0.5)
        left += share
        handles.append(Patch(facecolor=color, edgecolor="k", label=name))
    axA.set_xlim(0, 1)
    axA.set_ylim(-0.5, 0.5)
    axA.set_yticks([])
    axA.set_xlabel("Overall Relative VI")
    axA.set_title("Graph A. Predictor Groups", fontsize=11, style="italic")
    axA.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.55),
               ncol=2, fontsize=7.5, frameon=False)

    # Graph B: the ten predictors ranked, coloured by group
    pred_color = {p: c for _, members, c in GROUPS for p in members}
    s = rel.sort_values(ascending=True)
    axB.barh(s.index, s.values, color=[pred_color[p] for p in s.index],
             edgecolor="k", lw=0.5)
    for p, v in s.items():
        axB.text(v + 0.005, p, f"{v:.2f}", va="center", fontsize=7.5)
    axB.set_xlim(0, max(s.values) * 1.18)
    axB.set_xlabel("Overall Relative VI")
    axB.set_title("Graph B. Predictor Variables", fontsize=11, style="italic")

    fig.suptitle("Relative Variable Importance", fontsize=13, y=0.99)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(FIG5)
    plt.close(fig)
    return imp, rel


def main():
    plot_fig1()
    imp, rel = plot_fig5()

    print("Figure 1 -> " + os.path.relpath(FIG1, REPO))
    err = relative_errors()
    for bench in ("OLS_1Y_D", "OLS_5Y_M"):
        monthly, yearly = err[bench]
        print(f"  vs {bench}: mean monthly rel.err = {monthly.mean():+.3f}, "
              f"months LSTM wins = {int((monthly > 0).sum())}/120")

    print("\nFigure 5 -> " + os.path.relpath(FIG5, REPO))
    print(f"{'predictor':10s}{'VI (abs)':>12s}{'VI (rel)':>12s}")
    for p in rel.sort_values(ascending=False).index:
        print(f"{p:10s}{imp[p]:>12.4f}{rel[p]:>12.3f}")
    print("\ngroup shares (relative):")
    for name, members, _ in GROUPS:
        print(f"  {name:48s}{rel.reindex(members).fillna(0).sum():.3f}")


if __name__ == "__main__":
    main()
