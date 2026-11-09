# plot_fig4.py
# Fig. 4(a) of the paper -- false verification rate under benign MOBILITY, per location.
# Reads results/fig3b_fvr.csv.
#
# Every window here is the legitimate device: it enrolled in one environment and
# has since moved to the one on the x-axis. There is no attacker anywhere in the
# experiment, so every triggered verification counted is a wasted handshake.
#
# All methods are held to the same verification BUDGET by gen_fig3b.py,
# calibrated once in the enrollment environment and applied unchanged
# everywhere else, so the bars answer "same cost at home -- what does the move
# cost you?". Detection speed is measured at those points and checked to be
# comparable, so no method looks cheap merely by being deaf.
#
# The y axis is truncated (see --ymin): every bar sits near 0.6%, and a
# zero-based axis compresses the whole comparison into the top of the panel.
# Bar length is not proportional to value here.
#
#   python plot_fig4.py

import csv
import argparse

import numpy as np
import matplotlib.pyplot as plt

import plotstyle as ps

ORDER = ["Raw BFI", "Score threshold", "TrustBFI"]

# Legend text. Fig. 2 draws the gamma line on the panel, so the rule is visible
# there and the bare name is enough; here there is nothing to look at, so the
# bracket carries it. The rule rather than the bare symbol s(t): "score" is
# already in the name, and s(t) is the input TrustBFI uses too, so quoting it
# alone would suggest the two bars differ in their input rather than in what
# they do with it.
# "Score" is redundant once the bracket names the quantity, and dropping it
# buys enough width to run the legend at the full axis text size.
LABELS = {"Score threshold": "Threshold ($s<\\gamma$)"}


def load(path):
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            r["fvr_pct"] = float(r["fvr_pct"])
            r["score_loc"] = int(r["score_loc"])
            rows.append(r)
    return rows


def grouped_bars(ax, groups, methods, get, ylabel, fill=0.62):
    """One cluster per group, one bar per method. White fill, hatch-coloured
    outline, dense coloured diagonal hatch: the hatch direction is what
    separates the series in greyscale, and the white fill keeps the panel light.

    `fill` is the share of each unit slot the cluster occupies; the rest is the
    lane between clusters. Below ~0.7 the clusters read as separate groups
    rather than one continuous run of bars."""
    n = len(methods)
    w = fill / n
    gap = 0.012
    x = np.arange(len(groups))
    for i, m in enumerate(methods):
        vals = [get(m, g) for g in groups]
        pos = x + (i - (n - 1) / 2) * (w + gap)
        ps.draw_bars(ax, pos, vals, w, i, label=LABELS.get(m, m))
    ax.set_xticks(x)
    ax.set_ylabel(ylabel)
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=ps.repo("results", "fig3b_fvr.csv"))
    p.add_argument("--out", default="fig4")
    p.add_argument("--outdir", default=None)
    p.add_argument("--logy", action="store_true")
    # Eight clusters need room to stay separated, so this is wider than one
    # IEEE column by default. Whatever you set, set it to the width LaTeX will
    # actually GIVE the figure: the printed text size is the rcParam size times
    # (placed width / drawn width), and that ratio is what makes text too small.
    p.add_argument("--width", type=float, default=4.3)
    p.add_argument("--height", type=float, default=2.3)
    p.add_argument("--fontscale", type=float, default=1.0,
                   help="multiply every font size; for figures drawn small")
    # A rotated label is limited by the panel HEIGHT, and the legend by its
    # width -- both overrun once the fonts are sized for a half-column slot.
    p.add_argument("--ylabel", default="False verification rate (%)")
    p.add_argument("--legend-ncol", type=int, default=None)
    # Only locations 1 and 2 reach above ~0.9, so the upper right of the panel
    # is empty. Putting the legend there instead of above the axes removes two
    # text rows from the figure HEIGHT, which is the scarce direction in a
    # two-column letter.
    p.add_argument("--legend-inside", action="store_true")
    p.add_argument("--legend-loc", default="upper right")
    # An inside legend competes with the data for width, so it runs smaller
    # than the axis text rather than at the same size.
    p.add_argument("--legend-fs", type=float, default=0.78,
                   help="inside-legend text size, as a fraction of the axis "
                        "text size")
    # Headroom above the tallest bar. The default just keeps the bar off the
    # top spine; an inside legend needs somewhere to sit.
    p.add_argument("--headroom", type=float, default=None)
    # The bars all sit near 0.6%, so a zero-based axis compresses the whole
    # comparison into the top 5% of the panel. Truncating shows the variation
    # at the cost of bar LENGTH not being proportional to the value.
    p.add_argument("--ymin", type=float, default=0.5,
                   help="y-axis floor in %%; 0 for a conventional zero base")
    args = p.parse_args()

    ps.use(args.fontscale)
    rows = load(args.csv)
    methods = [m for m in ORDER if any(r["method"] == m for r in rows)]

    # only the cross-environment rows: the same-environment case is the control
    # and is ~0 for every method, so it carries no information here
    cross = [r for r in rows if r["condition"] == "cross"]
    groups = sorted({r["score_loc"] for r in cross})
    idx = {(r["method"], r["score_loc"]): r["fvr_pct"] for r in cross}

    fig, ax = plt.subplots(figsize=(args.width, args.height))
    grouped_bars(ax, groups, methods, lambda m, g: idx.get((m, g), 0.0),
                 args.ylabel)
    ax.set_xticklabels([str(g) for g in groups])
    ax.set_xlabel("Scoring location")
    if args.logy:
        ax.set_yscale("log")
    else:
        top = max(idx.values()) if idx else 1.0
        pad = args.headroom
        if pad is None:
            pad = 0.55 if args.legend_inside else 0.08
        ax.set_ylim(args.ymin, top + pad * (top - args.ymin))

    # after set_ylim, so "upper right" is measured against the final axes
    if args.legend_inside:
        # framealpha under 1 would let the grid lines show through the text
        ax.legend(loc=args.legend_loc, ncol=args.legend_ncol or 1,
                  fontsize=ps.fs(7.5) * args.legend_fs,
                  frameon=True, facecolor="white", edgecolor="none",
                  framealpha=0.95, handlelength=1.6, handletextpad=0.45,
                  borderpad=0.35, labelspacing=0.3).set_zorder(9)
    else:
        ps.legend_above(ax, ncol=args.legend_ncol)
    ps.tidy(ax, grid_axis="y", box=False)
    ax.grid(axis="x", visible=False)
    ps.save(fig, args.out, args.outdir)


if __name__ == "__main__":
    main()
