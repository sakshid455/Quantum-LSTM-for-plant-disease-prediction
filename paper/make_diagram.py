"""Pipeline + QLSTM gate diagram for the manuscript."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INK, MUTED = "#1f1f1e", "#6b6a64"
FILL = {"data": "#eef4fc", "cls": "#f3f2ec", "q": "#fdeee6", "out": "#e8f6f0"}
EDGE = {"data": "#2a78d6", "cls": "#8a897f", "q": "#eb6834", "out": "#1baf7a"}

fig, ax = plt.subplots(figsize=(11, 4.2))
ax.set_xlim(0, 110); ax.set_ylim(0, 42); ax.axis("off")

def box(x, y, w, h, text, kind, fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                fc=FILL[kind], ec=EDGE[kind], lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK)

def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.1, shrinkA=2, shrinkB=2))

# Inputs
box(1, 29, 17, 9, "Leaf phenotype history\n$t-3, \\ldots, t$\n(20 features / step)", "data")
box(1, 5, 17, 9, "Leaf image sequence\n$t-3, \\ldots, t$\n(visual variant)", "data")
box(22, 5, 14, 9, "ViT-B/16\n(frozen)\n768-d / step", "cls")
arrow(18, 9.5, 22, 9.5)
box(40, 17, 14, 9, "Standardise\n(train-fitted)\ninput $x_t$", "cls")
arrow(18, 33.5, 47, 26); arrow(36, 9.5, 47, 17)

# QLSTM cell block
ax.add_patch(FancyBboxPatch((58, 2), 34, 38, boxstyle="round,pad=0.3,rounding_size=1.5",
                            fc="white", ec=MUTED, lw=1, ls="--"))
ax.text(75, 37.6, "QLSTM cell (unrolled over T = 4)", ha="center", fontsize=8.5, color=INK, weight="bold")
gy = [28, 20, 12, 4]
for y, g in zip(gy, ["forget $f_t$", "input $i_t$", "output $o_t$", "candidate $g_t$"]):
    box(59.5, y, 8, 5.5, "MLP\nencoder", "cls", 6.5)
    box(69.5, y, 10, 5.5, "8-qubit VQC\n3 SEL layers", "q", 6.5)
    box(81.5, y, 9, 5.5, f"MLP\ndecoder\n{g}", "cls", 6)
    arrow(67.8, y + 2.75, 69.5, y + 2.75); arrow(79.8, y + 2.75, 81.5, y + 2.75)
arrow(54, 21.5, 59.3, 21.5)

# Heads
box(96, 24, 13, 8, "Disease head\nseverity $t+1$", "out")
box(96, 9, 13, 8, "Lesion head\narea $t+1$", "out")
arrow(92, 21, 96, 28); arrow(92, 21, 96, 13)
ax.text(102.5, 21, "$h_T$", ha="center", fontsize=8, color=MUTED)

fig.savefig("paper/figures/fig_architecture.png", dpi=300, bbox_inches="tight")
print("ok")
