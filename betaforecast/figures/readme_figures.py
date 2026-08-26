"""The figures the README embeds.

Each one has to carry its finding on its own -- a reader who never reads the
prose should still get the story -- so titles state the result rather than name
the axes, series are labelled on the marks, and the significant comparisons are
annotated in place.

Both figures are built from the published oos_forecasts.parquet alone, so they
regenerate in a fresh clone:

    python -m betaforecast.figures.readme_figures
"""
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from betaforecast.evaluation import evaluate
from betaforecast.importance import by_size
from betaforecast.paths import DOCS_IMG, OOS_FORECASTS, rel

# Categorical slots 1 and 2, and the chart ink, from the validated palette.
LSTM_C = "#2a78d6"
BENCH_C = "#eb6834"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_2,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": GRID,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _num(x, dp=2, signed=False):
    """Format with a typographic minus, so text matches the DM figures."""
    s = f"{x:+.{dp}f}" if signed else f"{x:.{dp}f}"
    return s.replace("-", "−")


def _headroom(ax, top=0.30, bottom=0.18):
    """Pad the y-range so annotations sit inside the axes, not over the title."""
    lo, hi = ax.get_ylim()
    span = hi - lo
    ax.set_ylim(lo - span * bottom, hi + span * top)


def _style(ax):
    ax.grid(True, axis="y", color=GRID, lw=0.8, ls="-")
    ax.set_axisbelow(True)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.8)


def horse_race(monthly):
    """Cumulative value-weighted loss differential against each benchmark.

    Two panels rather than two y-scales: the differentials differ by an order of
    magnitude, and putting them on one axis would flatten the interesting one.
    Plotted as benchmark minus LSTM, so up means the LSTM is more accurate.
    """
    idx = pd.PeriodIndex(monthly.index, freq="M").to_timestamp(how="end").normalize()
    cum_5ym = (monthly["OLS_5Y_M"] - monthly["LSTM"]).cumsum().to_numpy()
    cum_1yd = (monthly["OLS_1Y_D"] - monthly["LSTM"]).cumsum().to_numpy()
    covid = pd.Timestamp("2020-03-01")

    fig, axes = plt.subplots(2, 1, figsize=(9.2, 7.4), sharex=True,
                             gridspec_kw={"hspace": 0.38})

    for ax, cum, label in ((axes[0], cum_5ym, "5-year monthly beta"),
                           (axes[1], cum_1yd, "1-year daily beta")):
        _style(ax)
        ax.axhline(0, color=BASELINE, lw=1.0)
        ax.axvspan(covid, idx[-1], color=GRID, alpha=0.45, lw=0)
        ax.plot(idx, cum, color=LSTM_C, lw=2.0)
        ax.fill_between(idx, 0, cum, color=LSTM_C, alpha=0.10, lw=0)

    pk = int(np.argmax(cum_1yd))

    axes[0].set_title(
        "Against the five-year monthly beta, the LSTM pulls steadily ahead",
        fontsize=13.5, fontweight="bold", color=INK, loc="left", pad=10)
    # the peak year is read off the data, so the title cannot drift from it
    axes[1].set_title(
        f"Against the one-year daily beta, the edge builds to {idx[pk]:%Y}"
        " — then fades",
        fontsize=13.5, fontweight="bold", color=INK, loc="left", pad=10)

    _headroom(axes[0], top=0.16, bottom=0.04)
    _headroom(axes[1], top=0.30, bottom=0.18)

    # --- panel 1: the win, parked clear of the line -------------------------
    axes[0].text(0.985, 0.42,
                 f"+{cum_5ym[-1]:.1f} cumulative\nDM t = −6.55, p < 0.001",
                 transform=axes[0].transAxes, ha="right", va="top",
                 fontsize=10.5, color=INK_2, linespacing=1.45)

    # --- panel 2: the hump, its peak, and where it ends up ------------------
    axes[1].plot([idx[pk]], [cum_1yd[pk]], "o", ms=7, color=LSTM_C,
                 mec=SURFACE, mew=2, zorder=5)
    axes[1].annotate(f"peak {_num(cum_1yd[pk], signed=True)} ({idx[pk]:%b %Y})",
                     xy=(idx[pk], cum_1yd[pk]), xytext=(-14, 4),
                     textcoords="offset points", ha="right", va="bottom",
                     fontsize=10.5, color=INK_2)
    axes[1].annotate(f"ends at {_num(cum_1yd[-1], signed=True)}\n"
                     "DM t = 0.20, not significant",
                     xy=(idx[-1], cum_1yd[-1]), xytext=(-10, 30),
                     textcoords="offset points", ha="right", va="bottom",
                     fontsize=10.5, color=INK_2, linespacing=1.45)
    for ax in axes:
        ax.text(covid, ax.get_ylim()[1], " post-COVID origins", ha="left",
                va="top", fontsize=9.5, color=MUTED)
        ax.set_ylabel("Cumulative VW-MSE differential\n(↑ = LSTM more accurate)",
                      fontsize=10.5)
    axes[1].set_xlabel("Forecast origin", fontsize=10.5)

    fig.text(0.012, 0.012,
             "120 monthly out-of-sample origins, 2015–2024 · 10,145 stock-month "
             "forecasts · Diebold–Mariano, Newey–West lag 11",
             fontsize=9.5, color=MUTED)
    fig.subplots_adjust(left=0.115, right=0.985, top=0.925, bottom=0.115)
    return fig


def size_terciles(oos):
    """Where the accuracy actually differs: LSTM against the hard benchmark,
    within each size tercile.

    The five-year benchmark is left out on purpose -- it is beaten everywhere,
    and its much larger errors would compress the comparison that matters.
    """
    df = by_size.assign_terciles(oos.copy())
    df["cap"] = np.exp(df["ME"])
    vw = by_size.group_mse(df, weighted=True)
    dm = by_size.group_dm(df, weighted=True)

    groups = ["Small", "Mid", "Large"]
    ys = np.arange(len(groups))[::-1]                 # Small at the top

    fig, ax = plt.subplots(figsize=(9.2, 4.9))
    ax.grid(True, axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    for y, g in zip(ys, groups):
        a, b = vw.loc[g, "LSTM"], vw.loc[g, "OLS_1Y_D"]
        ax.plot([a, b], [y, y], color=MUTED, lw=1.4, zorder=1, alpha=0.55)
        ax.plot([b], [y], "o", ms=11, color=BENCH_C, mec=SURFACE, mew=2, zorder=3)
        ax.plot([a], [y], "o", ms=11, color=LSTM_C, mec=SURFACE, mew=2, zorder=3)

        p = dm.loc[g, "p_OLS_1Y_D"]
        gap = b - a                                   # > 0 means LSTM better
        verdict = (f"LSTM better by {gap:.3f}   p = {p:.3f}" if gap > 0
                   else f"LSTM worse by {abs(gap):.3f}   p = {p:.2f}, n.s.")
        ax.annotate(verdict, xy=(max(a, b), y), xytext=(16, 0),
                    textcoords="offset points", va="center", fontsize=10.5,
                    color=INK if gap > 0 else MUTED,
                    fontweight="bold" if gap > 0 else "normal")

    # direct labels on the top row only; the legend carries the rest
    ax.annotate("LSTM", xy=(vw.loc["Small", "LSTM"], ys[0]), xytext=(0, 17),
                textcoords="offset points", ha="center", fontsize=10.5,
                color=INK_2, fontweight="bold")
    ax.annotate("rolling OLS", xy=(vw.loc["Small", "OLS_1Y_D"], ys[0]),
                xytext=(0, 17), textcoords="offset points", ha="center",
                fontsize=10.5, color=INK_2, fontweight="bold")

    ax.set_yticks(ys)
    ax.set_yticklabels([f"{g}\ncaps" for g in groups], fontsize=11.5, color=INK)
    ax.set_xlabel("Value-weighted MSE within the tercile  (lower is better)",
                  fontsize=10.5)
    ax.set_xlim(0.062, 0.222)
    ax.set_ylim(-0.75, len(groups) - 0.15)

    ax.set_title("The LSTM's edge is in small and mid caps —\n"
                 "where a rolling regression is noisiest",
                 fontsize=14.5, fontweight="bold", color=INK, loc="left", pad=16)
    fig.text(0.012, 0.010,
             "Stocks sorted into terciles by market cap each month · value-weighted MSE "
             "within each tercile · 2015–2024\nThe five-year monthly beta is beaten in "
             "every tercile and is omitted here so the close comparison stays legible",
             fontsize=9.5, color=MUTED, linespacing=1.5)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.80, bottom=0.235)
    return fig


def report():
    os.makedirs(DOCS_IMG, exist_ok=True)
    oos = pd.read_parquet(OOS_FORECASTS)
    monthly = evaluate.monthly_vw_mse(evaluate.load())

    for name, fig in (("horse_race", horse_race(monthly)),
                      ("size_terciles", size_terciles(oos))):
        out = DOCS_IMG / f"{name}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
        plt.close(fig)
        print(f"wrote {rel(out)}")


if __name__ == "__main__":
    report()
