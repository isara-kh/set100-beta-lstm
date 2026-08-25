"""
Schematic diagram of a single LSTM memory cell (for illustration, not computation).
Canonical Olah-style cell: the cell-state conveyor C_{t-1} -> C_t across the top,
the forget / input / output gates regulating it, inputs h_{t-1} and x_t from the
left, and the hidden output h_t to the right. Serif fonts and a seaborn palette,
styled to match the companion network-architecture diagram (lstm_architecture.py).
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
C_CELL  = sns.color_palette("Blues", 6)[0]      # cell-boundary fill (very light)
C_GATE  = sns.color_palette("YlOrBr", 6)[2]     # sigmoid / tanh gate boxes
C_OP    = sns.color_palette("flare", 6)[2]      # pointwise operation circles
C_STATE = sns.color_palette("crest", 6)[3]      # cell-state line
C_HID   = "#33373d"                             # hidden-state line
EDGE    = "#33373d"

fig, ax = plt.subplots(figsize=(12.5, 6.6))
ax.set_xlim(0, 12.5)
ax.set_ylim(0, 6.6)
ax.axis("off")


def gate(x, y, label, name, w=0.95, h=0.95):
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.3, edgecolor=EDGE, facecolor=C_GATE, alpha=0.95, zorder=4))
    ax.text(x, y, label, ha="center", va="center", fontsize=15,
            fontweight="bold", color=EDGE, zorder=5)
    if name:
        ax.text(x, y - h/2 - 0.28, name, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=EDGE, zorder=5)


def op(x, y, sym, r=0.30):
    ax.add_patch(Circle((x, y), r, linewidth=1.3, edgecolor=EDGE,
                        facecolor=C_OP, alpha=0.95, zorder=4))
    ax.text(x, y, sym, ha="center", va="center", fontsize=15,
            fontweight="bold", color="white", zorder=5)


def line(pts, color, lw=2.2, alpha=0.9, head=True, z=2):
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        last = (x1, y1) == pts[-1]
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
            arrowstyle="-|>" if (head and last) else "-",
            mutation_scale=15, linewidth=lw, color=color, alpha=alpha,
            shrinkA=0, shrinkB=0, zorder=z))


# ---- cell boundary ----------------------------------------------------------
ax.add_patch(FancyBboxPatch((2.2, 1.05), 7.9, 4.5,
    boxstyle="round,pad=0.02,rounding_size=0.18",
    linewidth=1.6, edgecolor=EDGE, facecolor=C_CELL, alpha=0.45,
    linestyle=(0, (6, 4)), zorder=1))
ax.text(2.45, 5.27, "LSTM memory cell", ha="left", va="center",
        fontsize=10, style="italic", color=EDGE, zorder=5)

Y_C = 4.7      # cell-state conveyor height
Y_H = 1.6      # hidden-state line height
Y_G = 2.9      # gate row height

# ---- cell-state conveyor: C_{t-1} -> forget(x) -> input(+) -> C_t ------------
line([(0.55, Y_C), (3.3, Y_C)], C_STATE, head=False)
op(3.3, Y_C, r"$\times$")
line([(3.6, Y_C), (5.8, Y_C)], C_STATE, head=False)
op(5.8, Y_C, "+")
line([(6.1, Y_C), (9.1, Y_C)], C_STATE, head=False)
line([(9.1, Y_C), (12.0, Y_C)], C_STATE)
ax.text(0.5, Y_C + 0.33, r"$C_{t-1}$", ha="left", va="center",
        fontsize=14, color=EDGE, zorder=5)
ax.text(11.95, Y_C + 0.33, r"$C_{t}$", ha="right", va="center",
        fontsize=14, color=EDGE, zorder=5)
ax.text(6.15, Y_C + 0.30, "cell state (long-run memory)", ha="left",
        va="center", fontsize=8.5, style="italic", color=EDGE, zorder=5)

# ---- gates ------------------------------------------------------------------
gate(3.3, Y_G, r"$\sigma$", "Forget")
gate(5.3, Y_G, r"$\sigma$", "Input")
gate(6.3, Y_G - 0.0, r"$\tanh$", "Candidate", w=1.05)
gate(8.6, Y_G, r"$\sigma$", "Output")

# gate -> conveyor wiring
line([(3.3, Y_G + 0.48), (3.3, Y_C - 0.30)], C_HID, lw=1.6)            # forget -> x
op(5.8, 3.7, r"$\times$", r=0.26)                                      # input gate * candidate
line([(5.3, Y_G + 0.48), (5.55, 3.7 - 0.0)], C_HID, lw=1.6, head=False)
line([(6.3, Y_G + 0.52), (6.05, 3.7)], C_HID, lw=1.6, head=False)
line([(5.8, 3.96), (5.8, Y_C - 0.30)], C_HID, lw=1.6)                  # -> +

# ---- output path: C_t -> tanh -> (x output gate) -> h_t ---------------------
op(8.6, 4.7, r"$\tanh$", r=0.34)                                       # tanh on cell state
line([(9.1, Y_C), (9.1, 4.7), (8.94, 4.7)], C_STATE, lw=1.6, head=False)
op(8.6, 3.85, r"$\times$", r=0.26)                                     # output * tanh(C)
line([(8.6, 4.36), (8.6, 4.11)], C_STATE, lw=1.6, head=False)
line([(8.6, Y_G + 0.48), (8.6, 3.85 - 0.26)], C_HID, lw=1.6, head=False)
line([(8.6, 3.59), (8.6, Y_H), (12.0, Y_H)], C_HID)                    # -> h_t out

# ---- inputs: h_{t-1} and x_t in from the left, fan to all gates -------------
line([(0.55, Y_H), (2.2, Y_H)], C_HID, head=False)
ax.text(0.5, Y_H, r"$h_{t-1}$", ha="left", va="center",
        fontsize=14, color=EDGE, zorder=5)
ax.text(0.5, Y_H - 0.55, r"$x_t$  (10 predictors)", ha="left", va="center",
        fontsize=11, color=EDGE, zorder=5)
line([(1.5, Y_H - 0.55), (1.5, Y_H), (2.2, Y_H)], C_HID, head=False, lw=1.6)
# fan-in node then to each gate
ax.add_patch(Circle((2.5, Y_H), 0.06, color=EDGE, zorder=5))
for gx in (3.3, 5.3, 6.3, 8.6):
    line([(2.5, Y_H), (gx, Y_H), (gx, Y_G - 0.48)], C_HID,
         lw=1.0, alpha=0.55, head=False)

# h_t output label
ax.text(11.95, Y_H + 0.33, r"$h_{t}$", ha="right", va="center",
        fontsize=14, color=EDGE, zorder=5)
ax.text(11.95, Y_H - 0.40, "hidden output", ha="right", va="center",
        fontsize=8.5, style="italic", color=EDGE, zorder=5)

plt.tight_layout(pad=0.3)
out_pdf = "figures/lstm_cell.pdf"
out_png = "figures/lstm_cell_preview.png"
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print("wrote", out_pdf, "and", out_png)
