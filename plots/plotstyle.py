# plotstyle.py
# Shared figure style for the TrustBFI plots, so the plot scripts do not each
# carry their own copy of the rcParams and the palette.
#
# Matched to the SPARK paper's figure style so the group's papers read as a
# series: saturated red / black / blue / magenta, a full box frame, sans-serif
# labels, and the legend in one horizontal row above the panel.
#
# The "our method" slot is magenta with an up-triangle, the same slot SPARK
# occupies in its own figures -- so TrustBFI is magenta here, always.
#
# This palette is not colourblind-safe on hue alone (red/black and blue/magenta
# are the risky pairs). Every series therefore carries a distinct MARKER and a
# distinct LINE STYLE as well, which is what makes it readable in greyscale
# print and under colour-vision deficiency. Keep both when adding a series.

import os

import matplotlib as mpl
import matplotlib.pyplot as plt

# These scripts live one level down from the repo root, so every default path
# is resolved against the root rather than the working directory. Without this
# the scripts only run from the root, which is a trap when the obvious thing to
# do is cd into this folder and run one.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def repo(*parts):
    """Absolute path to something in the repository, from anywhere."""
    return os.path.join(ROOT, *parts)

RED = "#e8262a"        # Dense 2D-MUSIC slot
BLACK = "#000000"      # Coarse 2D-MUSIC slot
BLUE = "#1f2fd6"       # 2D-ESPRIT slot
MAGENTA = "#e663e6"    # SPARK slot -> our method, for markers and bars
# The light magenta is tuned for filled markers and bars. As a thin continuous
# line it washes out against white, so trace plots use this deeper tone -- same
# hue family, so TrustBFI keeps its identity across figures.
MAGENTA_DARK = "#a3179b"

INK = "#000000"
INK2 = "#333333"
MUTED = "#8a8a8a"
GRID = "#dddddd"

# Event accent: substitution markers and annotations in the Fig. 2 trace. Kept
# as the red slot so events read in the same hue family as the figures' warm
# series rather than introducing a fifth colour.
ORANGE = RED

# entity -> (colour, marker, linestyle). Colour follows the ENTITY, never its
# rank, so a series keeps its identity across every figure.
SERIES = {
    "TrustBFI":      (MAGENTA, "^", "-."),
    "Periodic":      (RED, "D", "-"),
    "Always-Verify": (BLACK, "v", "--"),
    "Age-Only":      (BLACK, "v", "--"),
    # One name for the memoryless baseline: threshold s(t) against gamma, no
    # belief and no memory.
    "Score threshold": (BLUE, "s", ":"),
    "Raw BFI":         (BLUE, "s", ":"),
    "Score threshold (oracle)": (BLUE, "s", (0, (1, 3))),
}

# Hatched bar scheme, matching the group's BER figures: a WHITE fill and a
# dense saturated diagonal hatch. The hatch colour and direction are what
# separate the series -- the fill stays white so the bars stay light on the
# page and value labels remain readable over them.
#
# Matplotlib < 3.11 has no per-artist hatch colour: the hatch takes the
# artist's edgecolor, so the outline is drawn in the hatch colour too. That is
# deliberate here -- a black outline boxes the bars in and fights the hatch.
# (hatch colour, hatch pattern)
BARS = [
    ("#b03a55", "/////"),    # crimson -> first series (baseline)
    ("#2e6aa8", "\\\\\\\\\\"),  # blue    -> second series (ours)
    ("#8a7530", "xxxxx"),    # sand    -> third, if ever needed
]

BAR_FACE = "white"


def bar_style(i):
    """(hatch colour, hatch pattern) for the i-th bar series."""
    return BARS[i % len(BARS)]


def draw_bars(ax, pos, vals, width, i, label=None, **kw):
    """One hatched bar series."""
    hc, hatch = bar_style(i)
    ax.bar(pos, vals, width, facecolor=BAR_FACE, edgecolor=hc, hatch=hatch,
           linewidth=0.8, label=label, zorder=3, **kw)


# IEEE two-column: 3.5 in is one full column. HALF_W is one of two
# subfigures side by side inside a single column.
COL_W = 3.5
HALF_W = 1.68

# What actually sets the printed text size is the ratio between the figure's
# drawn width and the width LaTeX gives it. A 3.5 in figure dropped into a
# 1.7 in subfigure slot is scaled to 49%, so an 8 pt label prints at 3.9 pt,
# and no amount of font bumping fixes it while the figure is drawn twice as
# wide as it is placed. Draw at the
# placed width, or span both columns with figure*.
#
# SCALE exists for the residual case where a figure is deliberately drawn small
# and its text needs to grow relative to the axes. use() sets it; fs() applies
# it to the one-off font sizes scattered through the plot scripts, which
# rcParams cannot reach.
SCALE = 1.0


def fs(pt):
    """A one-off font size in points, scaled with the rest of the figure."""
    return pt * SCALE


def use(scale=1.0, axis_scale=1.0):
    """scale multiplies every font size; axis_scale then multiplies the AXIS
    text on top of it -- tick labels and axis labels only.

    The two are separate because the axis text and the in-panel annotations do
    not want the same size. Annotations sit among the data and have to stay out
    of its way; axis text sits outside it and is what a reader squints at
    first. Raising `scale` alone to make the ticks legible drags the
    annotations up with it until they collide."""
    global SCALE
    SCALE = scale
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 8 * scale,
        "axes.labelsize": 8.5 * scale * axis_scale,
        "axes.titlesize": 8.5 * scale,
        "legend.fontsize": 7.5 * scale,
        "xtick.labelsize": 7.5 * scale * axis_scale,
        "ytick.labelsize": 7.5 * scale * axis_scale,
        "axes.edgecolor": BLACK,
        "axes.linewidth": 0.8,
        "xtick.color": BLACK,
        "ytick.color": BLACK,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "text.color": INK,
        "axes.labelcolor": INK,
        "hatch.linewidth": 0.8,
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        "errorbar.capsize": 2.5,
        "figure.dpi": 200,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def tidy(ax, grid_axis="both", box=True):
    """Faint grid behind the data, with a full box frame as in the SPARK
    figures. box=False drops the top and right spines (and their ticks) for an
    open L-frame, which suits the hatched bar figures -- a closed box around
    already-outlined bars reads as clutter."""
    for name, sp in ax.spines.items():
        sp.set_visible(box or name in ("left", "bottom"))
    if not box:
        ax.tick_params(top=False, right=False)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def legend_above(ax, ncol=None, y=1.02, handles=None, labels=None, **kw):
    """One horizontal row above the panel, no frame -- the SPARK layout."""
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    # frameon is a default here, not a fixed choice: the hatched-bar figures
    # want a boxed legend, so callers must be able to override it
    kw.setdefault("frameon", False)
    ax.legend(handles, labels, loc="lower center",
              bbox_to_anchor=(0.5, y), ncol=ncol or len(handles),
              handlelength=1.6, columnspacing=1.1, handletextpad=0.45,
              borderpad=0.2, **kw)


def style(name):
    """(colour, marker, linestyle) for a series name."""
    return SERIES.get(name, (MUTED, "o", "-"))


def save(fig, stem, outdir=None):
    """Write both PNG (for review) and PDF (for LaTeX)."""
    outdir = outdir or repo("figs")
    os.makedirs(outdir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{outdir}/{stem}.{ext}")
    plt.close(fig)
    print(f"[out] {outdir}/{stem}.png  {outdir}/{stem}.pdf")
