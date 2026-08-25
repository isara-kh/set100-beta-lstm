"""Walk-forward (rolling-window) evaluation diagram (seaborn, light pastel).

Schematic that illustrates the out-of-sample backtesting scheme. Rows = the ten
walk-forward iterations. Each iteration trains once on a 5-year block (then 1y
validation, 1y gap) and forecasts the 12 month-ends of a 1-year test block; the
window shifts forward one year per iteration, so the test blocks tile the
2015-2024 out-of-sample period (10 x 12 = 120 month-ends).
"""
import os

from betaforecast.paths import REPO

os.environ.setdefault("MPLCONFIGDIR", str(REPO / ".cache_mpl"))

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

sns.set_theme(style="white")
pastel = sns.color_palette("pastel")
C_TRAIN, C_VAL, C_GAP, C_TEST = pastel[0], pastel[2], "0.86", pastel[3]

n_iter = 10
row_h, dy = 0.60, 1.0
fig, ax = plt.subplots(figsize=(8.6, 5.2))

for k in range(n_iter):
    y = (n_iter - 1 - k) * dy
    ts = 2008 + k
    blocks = [(ts, 5, C_TRAIN, None), (ts + 5, 1, C_VAL, None),
              (ts + 6, 1, C_GAP, "////"), (ts + 7, 1, C_TEST, None)]
    for x0, w, c, hatch in blocks:
        ax.add_patch(mpatches.Rectangle((x0, y), w, row_h, fc=c, ec="0.4",
                                        lw=0.8, hatch=hatch, zorder=2))
    ax.text(2007.5, y + row_h / 2, f"Iter {k+1}", ha="right", va="center", fontsize=8.5)
    if k == 0:                                   # label the blocks once, on the top row
        ax.text(ts + 2.5, y + row_h / 2, "train (5y)", ha="center", va="center", fontsize=8.5)
        for lx, lab in [(ts + 5.5, "val"), (ts + 6.5, "gap"), (ts + 7.5, "test")]:
            ax.text(lx, y + row_h + 0.10, lab, ha="center", va="bottom", fontsize=7.5)

# callout: each test block = 12 monthly forecasts
yk1 = (n_iter - 1 - 1) * dy + row_h / 2
ax.annotate("each test year =\n12 monthly $\\beta$ forecasts", xy=(2016.5, yk1),
            xytext=(2019.2, yk1 + 1.4), fontsize=8.2, color="0.25", ha="left",
            arrowprops=dict(arrowstyle="->", color="0.5", lw=0.9))

# out-of-sample bracket under the tiled test blocks (2015-2025)
yb = -0.55
ax.annotate("", xy=(2015, yb), xytext=(2025, yb),
            arrowprops=dict(arrowstyle="<->", color="0.4", lw=1.1))
ax.text(2020, yb - 0.18, "out-of-sample: 2015 to 2024  (120 month-ends, $10\\times12$)",
        ha="center", va="top", fontsize=8.5, color="0.2")

# "shifts one year" indicator, placed in the empty lower-left region
ax.annotate("window shifts\nforward one year", xy=(2014.0, 3.35),
            xytext=(2008.8, 1.7), fontsize=8.0, color="0.3", ha="center",
            arrowprops=dict(arrowstyle="->", color="0.5", lw=0.9))

ax.set_xlim(2006.6, 2025.8)
ax.set_ylim(yb - 0.9, (n_iter - 1) * dy + row_h + 0.7)
ax.set_yticks([])
ax.set_xticks(range(2008, 2026, 2))
ax.set_xticklabels(range(2008, 2026, 2), fontsize=8.5)
ax.tick_params(axis="x", length=0)
ax.set_xlabel("year")
sns.despine(left=True)
fig.tight_layout()
out = os.path.join(REPO, "figures", "walk_forward.pdf")
fig.savefig(out, bbox_inches="tight")
fig.savefig(os.path.join(REPO, "figures", "walk_forward_preview.png"), dpi=140, bbox_inches="tight")
print(f"wrote {os.path.relpath(out, REPO)} and walk_forward_preview.png")
