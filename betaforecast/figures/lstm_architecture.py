"""
Schematic diagram of the LSTM network architecture used in this study.
Left-to-right block style, rendered colourfully with a seaborn palette and serif
fonts. This is a hand-drawn illustration of the data flow, not executable model code.

Pipeline:  SET100 panel -> 4 predictor groups (10 predictors) ->
           input layer x1..x10 (12-month lookback) -> LSTM layer (8 units) ->
           dense layer (1 unit, linear) -> one-year-ahead realized-beta forecast.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import seaborn as sns

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "cm",
})

# ---- palette ----------------------------------------------------------------
pal = sns.color_palette("Set2", 8)
C_RAW   = sns.color_palette("Blues", 6)[2]
C_GRP   = [pal[0], pal[1], pal[2], pal[3]]          # 4 predictor groups
C_INPUT = sns.color_palette("crest", 6)[3]
C_LSTM  = sns.color_palette("flare", 6)[2]
C_DENSE = sns.color_palette("YlOrBr", 6)[3]
C_OUT   = sns.color_palette("Purples", 6)[3]
EDGE    = "#33373d"
LINE    = "#9aa0a8"

fig, ax = plt.subplots(figsize=(14, 7.2))
ax.set_xlim(0, 14)
ax.set_ylim(0, 7.6)
ax.axis("off")

def rbox(x, y, w, h, color, text, fs=11, weight="bold", tcolor=EDGE):
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.3, edgecolor=EDGE, facecolor=color, alpha=0.95, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            fontweight=weight, color=tcolor, zorder=4)

def node(x, y, r, color, text, fs=10):
    ax.add_patch(Circle((x, y), r, linewidth=1.2, edgecolor=EDGE,
                         facecolor=color, alpha=0.95, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            fontweight="bold", color=EDGE, zorder=4)

def arrow(x0, y0, x1, y1, color=LINE, lw=0.7, alpha=0.55, head=False, z=1):
    style = "-|>" if head else "-"
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
        mutation_scale=12, linewidth=lw, color=color, alpha=alpha, zorder=z))

def caption(x, y, lines, fs=11):
    ax.text(x, y, "\n".join(lines), ha="center", va="top",
            fontsize=fs, fontweight="bold", color=EDGE)

YC = 4.1                      # vertical centre of the network
X_RAW, X_GRP, X_INP, X_LSTM, X_DENSE, X_OUT = 1.15, 3.4, 5.9, 8.6, 11.0, 12.9

# ---- column 1: raw panel ----------------------------------------------------
rbox(X_RAW, YC, 1.5, 2.0, C_RAW,
     "SET100\nStock–Month\nPanel", fs=11)
caption(X_RAW, 1.6, ["Sample", "(monthly)"])

# ---- column 2: four predictor groups ---------------------------------------
groups = [
    ("Historical Beta",        3),
    ("Accounting",             2),
    ("Technical & Liquidity",  4),
    ("Macroeconomic",          1),
]
gy = [6.45, 5.15, 3.55, 2.05]
for (name, n), yy, col in zip(groups, gy, C_GRP):
    txt = name + "\n(%d)" % n
    rbox(X_GRP, yy, 1.95, 0.95, col, txt, fs=9.0, weight="bold")
    arrow(X_RAW + 0.78, YC + (yy - YC) * 0.18, X_GRP - 1.0, yy,
          color=EDGE, lw=1.1, alpha=0.8, head=True, z=2)
caption(X_GRP, 1.15, ["10 Predictors", "(4 groups)"])

# ---- column 3: input layer x1..x10 -----------------------------------------
xlabels = [r"$x_{%d}$" % k for k in range(1, 11)]
iy = [6.7 - k * (6.7 - 1.5) / 9 for k in range(10)]
for yy, lab in zip(iy, xlabels):
    node(X_INP, yy, 0.27, C_INPUT, lab, fs=9)
# groups -> input fan
gy_full = []
for (name, n), yy in zip(groups, gy):
    gy_full += [yy] * n
for yy, ny in zip(gy_full, iy):
    arrow(X_GRP + 0.95, yy, X_INP - 0.28, ny, color=LINE, lw=0.6, alpha=0.45)
caption(X_INP, 0.95, ["Input Layer", "(12-month lookback)"])

# ---- column 4: LSTM layer (8 units) ---------------------------------------
lstm_labels = ["LSTM 1", "LSTM 2", "LSTM 3", "LSTM 4", "LSTM 5", "LSTM 8"]
ly = [6.55, 5.55, 4.55, 3.55, 2.55, 1.25]
for yy, lab in zip(ly, lstm_labels):
    rbox(X_LSTM, yy, 1.35, 0.62, C_LSTM, lab, fs=9.5)
ax.text(X_LSTM, 1.92, r"$\vdots$", ha="center", va="center",
        fontsize=16, color=EDGE, zorder=4)
# input -> LSTM full fan
for ny in iy:
    for ly_ in ly:
        arrow(X_INP + 0.28, ny, X_LSTM - 0.68, ly_, color=LINE, lw=0.4, alpha=0.30)
caption(X_LSTM, 0.55, ["LSTM Layer", "(8 units)"])

# ---- column 5: dense layer (1 unit, linear) --------------------------------
node(X_DENSE, YC, 0.46, C_DENSE, "1", fs=12)
for ly_ in ly:
    arrow(X_LSTM + 0.68, ly_, X_DENSE - 0.46, YC, color=LINE, lw=0.6, alpha=0.5)
caption(X_DENSE, 2.7, ["Dense Layer", "(1 unit, linear)"])

# ---- column 6: output -------------------------------------------------------
arrow(X_DENSE + 0.46, YC, X_OUT - 0.05, YC, color=EDGE, lw=1.6, alpha=0.9, head=True, z=2)
ax.text(X_OUT + 0.05, YC, r"$\widehat{\beta}^{\,\mathrm{LSTM}}_{i,\,t+12\,|\,t}$",
        ha="left", va="center", fontsize=15, fontweight="bold", color=EDGE)
caption(X_OUT + 0.45, 2.9, ["One-year-ahead", "realized-beta", "forecast"])

plt.tight_layout(pad=0.4)
out_pdf = "figures/lstm_dataflow.pdf"
out_png = "figures/lstm_dataflow_preview.png"
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print("wrote", out_pdf, "and", out_png)
