"""Shared figure style so every chart in figures/ looks consistent."""
import matplotlib

matplotlib.use("Agg")   # draw to files; no window needed
import matplotlib.pyplot as plt

# Validated categorical order (blue, orange, aqua). Text never uses series colours.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# Sequential blues, light (short maturity) -> dark (long maturity).
BLUES = ["#b7d3f2", "#94bdec", "#6fa5e4", "#4a8ddb", "#2a78d6", "#1f63b6", "#174f93", "#103c71", "#0a2a51"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 10, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "lines.linewidth": 2.0, "legend.frameon": False, "legend.fontsize": 9,
    "font.size": 10, "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
})


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
