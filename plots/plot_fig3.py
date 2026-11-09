# plot_fig3.py
# Fig. 3 of the paper -- cost of a security level under attack: 5% and 10% substitution
# rates as two panels of ONE figure.
#
# Drawn as a single full-column figure rather than two subfigures, because two
# separate figures side by side each get half a column, and everything in them
# is then scaled to ~49%, which makes the text unreadable. One figure
# at \columnwidth is placed at the size it was drawn, so the text prints at the
# size it was set.
#
# A single figure also saves space: the legend, the y-axis label and the
# y tick labels appear ONCE instead of twice, and that recovered width goes to
# the data. The two panels share a y-axis, which is not just tidiness -- panels
# on different scales cannot be compared by eye, and comparing them is the only
# reason to put them next to each other.
#
#   python plot_fig3.py
#
import argparse

import matplotlib.pyplot as plt

import plotstyle as ps
from plot_fig2 import load, draw_panel, ORDER

# matplotlib text is not LaTeX -- "%" needs no backslash here, and one would
# be drawn literally
# Short tags: at the font sizes this figure is set in, "(a) 5% attack rate"
# is wider than its panel and runs into the neighbouring one. The word "rate"
# belongs in the caption.
PANELS = [("results/fig3a_attack5.csv", "(a) 5% attack"),
          ("results/fig3a_attack10.csv", "(b) 10% attack")]

# Three targets, not the nine of fig2: past ~25 windows the policies converge
# under attack, and at this panel width more points only crowd the ratio
# labels. The trend needs its ends and middle, nothing else.
TARGETS = [1.0, 5.0, 25.0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="fig3")
    p.add_argument("--outdir", default=None)
    p.add_argument("--width", type=float, default=3.5,
                   help="one IEEE column; the figure is placed at this width")
    p.add_argument("--height", type=float, default=2.2)
    p.add_argument("--fontscale", type=float, default=1.0)
    p.add_argument("--no-ratios", action="store_true")
    # Figure coordinates. The legend hangs off the FIGURE so it can centre over
    # both panels, which means it does not track the axes the way an axes-level
    # legend does -- at 1.0 it floats well clear of the panel titles.
    p.add_argument("--legend-y", type=float, default=0.92)
    p.add_argument("--ratio-dy", type=float, default=7.0,
                   help="lift the Nx labels, in points, off the lower marker")
    # Padding above the top series and below the bottom one, as multipliers on
    # a log axis. The default leaves a lot of blank sky above the
    # Always-Verify line; tightening it is free height.
    p.add_argument("--ypad-top", type=float, default=1.3)
    p.add_argument("--ypad-bottom", type=float, default=0.7)
    # Figure fraction, so it does NOT track the tick labels: the shorter the
    # figure, the closer the ticks come to the bottom edge and the further
    # down this has to sit to clear them.
    p.add_argument("--xlabel-y", type=float, default=-0.12)
    args = p.parse_args()

    ps.use(args.fontscale)
    fig, axes = plt.subplots(1, 2, figsize=(args.width, args.height),
                             sharey=True, gridspec_kw=dict(wspace=0.10))

    spans = []
    for ax, (csv_rel, title) in zip(axes, PANELS):
        rows = load(ps.repo(*csv_rel.split("/")))
        methods = [m for m in ORDER if any(r["policy"] == m for r in rows)]
        feasible, av = draw_panel(ax, rows, TARGETS, methods,
                                  show_ratios=not args.no_ratios,
                                  ratio_dy=args.ratio_dy)
        # av must be in the range, or the Always-Verify line is drawn off the
        # top of the axis and silently disappears
        spans += feasible + ([av] if av else [])
        # Tag above the panel, not below: below is where the shared x-axis
        # label goes, and the two collide. The text can still say "Fig. 3(a)".
        ax.set_title(title, fontsize=ps.fs(8), pad=3)

    # one y range for both, or the panels are not comparable by eye
    axes[0].set_ylim(min(spans) * args.ypad_bottom,
                     max(spans) * args.ypad_top)
    axes[0].set_ylabel("Verifications\nper 1k comm. rounds")

    # shared x-axis label, centred under the pair
    # Negative y puts it BELOW the tick labels. savefig(bbox="tight") grows
    # the canvas to include it, so nothing is cropped.
    fig.supxlabel("Allowed detection delay (comm. rounds)",
                  fontsize=ps.fs(8.5),
                  y=args.xlabel_y)
    # One legend for the whole figure, not one per panel. It hangs off the
    # FIGURE rather than an axis so it centres over both panels; anchoring it
    # to axes[0] would centre it over the left panel only.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, args.legend_y),
               ncol=len(handles), frameon=False, numpoints=1,
               handlelength=1.6, columnspacing=1.1, handletextpad=0.45,
               borderpad=0.2)
    ps.save(fig, args.out, args.outdir)


if __name__ == "__main__":
    main()
