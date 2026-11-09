# plot_fig2.py
# Cost of a security level, attack-free traffic (not in the paper).
# Reads results/fig3a_tradeoff.csv.
#
# plot_fig3.py reuses draw_panel() below for its two attack-rate panels;
# keep that function's signature stable.
#
#   x : tolerated exposure  (windows a substitution may survive undetected)
#   y : strong verifications per 1000 windows needed to hold that exposure
#
# At each x every policy answers the SAME question, so the markers are directly
# comparable and the vertical connector between them is a real like-for-like
# gap. Lower is better.
#
# For each policy and each x we take the CHEAPEST setting of its own parameter
# whose measured exposure is at or below x. A policy that cannot reach a given
# exposure at any setting is drawn as an open marker near the top of the axis.
#
# Markers only. A policy's points across different x are separate operating
# conditions, not samples of a continuous signal, so they are not joined.
#
#   python plot_fig2.py

import csv
import argparse

import numpy as np
import matplotlib.pyplot as plt

import plotstyle as ps

# Age-Only fires on elapsed time alone, which is the same rule as Periodic, so
# it adds a third marker that tracks the second one and carries no argument the
# figure is making. gen_fig3a.py still sweeps it and the CSV still holds it --
# pass --show-ageonly to put it back.
ORDER = ["Periodic", "TrustBFI"]
DEFAULT_TARGETS = [1, 2, 5, 10, 25, 50, 100, 250, 500]


def load(path):
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append(dict(policy=r["policy"], param=float(r["param_value"]),
                             # TOTAL handshakes: the cost actually paid.
                             # Plotting "wasted only" needs detections counted
                             # once per episode, not once per failed handshake;
                             # otherwise it inflates the ratio badly.
                             ovh=float(r["overhead_per_1k"]),
                             total=float(r["overhead_per_1k"]),
                             det=float(r.get("detect_per_1k") or 0.0),
                             exp=float(r["exposure_mean"]),
                             missed=float(r["missed_rate"])))
    return rows


def cheapest(rows, policy, target):
    """Min-overhead setting of `policy` whose exposure is <= target."""
    ok = [r for r in rows if r["policy"] == policy and r["exp"] <= target]
    return min(ok, key=lambda r: r["ovh"]) if ok else None


def draw_panel(ax, rows, targets, methods, show_ratios=True, ratio_dy=0.0):
    """Draw the cost-vs-delay panel on `ax`.

    Split out of main() so plot_fig3.py can call it twice into one figure with
    a shared legend and y-axis. Returns (feasible y values, Always-Verify
    level) so a caller drawing several panels can give them a common y range --
    two panels on different scales would not be comparable by eye, which is the
    entire point of putting them side by side.

    ratio_dy separates the Nx labels, in points. They sit at the geometric mean
    of the pair, which is the visual midpoint of the connector on a log axis --
    but the rightmost target has the shortest connector, so its label lands
    close to its neighbour. The rightmost one is therefore pushed DOWN by
    ratio_dy and every other one UP, which doubles the gap between the last two
    without needing to know which index is which.
    """
    xpos = {t: i for i, t in enumerate(targets)}
    av_ovh = min((r["ovh"] for r in rows if r["policy"] == "Always-Verify"),
                 default=None)
    sel = {(m, t): cheapest(rows, m, t) for m in methods for t in targets}
    feasible = [v["ovh"] for v in sel.values() if v]
    ceiling = max(feasible + ([av_ovh] if av_ovh else [1.0])) * 1.8

    # like-for-like gap at each tolerated exposure
    for t in targets:
        ys = [sel[(m, t)]["ovh"] for m in methods if sel[(m, t)]]
        if len(ys) > 1:
            ax.plot([xpos[t]] * 2, [min(ys), max(ys)], color=ps.MUTED, lw=0.9,
                    alpha=0.55, zorder=2, solid_capstyle="round")

    for m in methods:
        c, mk, _ = ps.style(m)
        hero = (m == "TrustBFI")
        xs = [xpos[t] for t in targets if sel[(m, t)]]
        ys = [sel[(m, t)]["ovh"] for t in targets if sel[(m, t)]]
        ax.plot(xs, ys, linestyle="none", marker=mk, color=c,
                ms=ps.fs(7 if hero else 5.5), mec="white", mew=0.8, label=m,
                zorder=6 if hero else 4)
        # exposures this policy cannot reach at any setting
        bad = [xpos[t] for t in targets if not sel[(m, t)]]
        if bad:
            ax.plot(bad, [ceiling * 0.85] * len(bad), linestyle="none",
                    marker=mk, color=c, ms=ps.fs(6), mfc="none", mew=1.1,
                    zorder=5)

    if av_ovh is not None:
        c, _, _ = ps.style("Always-Verify")
        ax.axhline(av_ovh, color=c, lw=1.1, ls=(0, (4, 2)), zorder=3,
                   label="Always-Verify")

    # how many times cheaper TrustBFI is at the same tolerated exposure
    for t in targets:
        tb, pe = sel.get(("TrustBFI", t)), sel.get(("Periodic", t))
        if tb and pe and tb["ovh"] > 0 and show_ratios:
            ratio = pe["ovh"] / tb["ovh"]
            # sub-1 ratios would round to "0x", which reads as "no advantage"
            # when it actually means TrustBFI costs MORE here
            rtxt = f"{ratio:.0f}$\\times$" if ratio >= 10 else f"{ratio:.1f}$\\times$"
            # the rightmost label would hang off the axis, and worse at larger
            # font scales, so it goes on the inside of its connector instead
            last = (t == targets[-1])
            ax.annotate(rtxt,
                        xy=(xpos[t], float(np.sqrt(tb["ovh"] * pe["ovh"]))),
                        xytext=(-3 if last else 3,
                                -ratio_dy if last else ratio_dy),
                        textcoords="offset points",
                        fontsize=ps.fs(7.5), color=ps.INK,
                        ha="right" if last else "left", va="center",
                        zorder=7,
                        bbox=dict(boxstyle="round,pad=0.12", fc="white",
                                  ec="none", alpha=0.75))

    ax.set_xticks(list(xpos.values()))
    ax.set_xticklabels([f"{t:g}" for t in targets])
    ax.set_xlim(-0.5, len(targets) - 0.5)
    ax.set_yscale("log")
    ps.tidy(ax, grid_axis="y")
    ax.grid(axis="x", visible=False)
    return feasible, av_ovh


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=ps.repo("results", "fig3a_tradeoff.csv"))
    p.add_argument("--out", default="fig2")
    p.add_argument("--outdir", default=None)
    p.add_argument("--width", type=float, default=3.5,
                   help="drawn width in inches. Set this to the width LaTeX "
                        "will give the figure: scaling a wide figure down is "
                        "what makes the text unreadable.")
    p.add_argument("--height", type=float, default=2.85)
    p.add_argument("--fontscale", type=float, default=1.0,
                   help="multiply every font size; for figures drawn small")
    p.add_argument("--targets", nargs="+", type=float, default=DEFAULT_TARGETS)
    p.add_argument("--ylabel",
                   default="Verifications needed\nper 1k comm. rounds",
                   help="the two-line default costs width; pass a one-line "
                        "form for figures placed at half a column")
    p.add_argument("--xlabel",
                   default="Allowed detection delay (comm. rounds)",
                   help="the default overruns the width at half-column scale")
    p.add_argument("--legend-ncol", type=int, default=None,
                   help="legend columns; 3 entries on one row overrun a "
                        "half-column figure, so wrap to 2 there")
    p.add_argument("--no-ratios", action="store_true",
                   help="drop the Nx annotations, which collide once the "
                        "fonts are large enough for a half-column placement")
    p.add_argument("--show-ageonly", action="store_true",
                   help="also plot Age-Only, the pure elapsed-time policy")
    args = p.parse_args()

    ps.use(args.fontscale)
    order = ORDER if not args.show_ageonly else ["Periodic", "Age-Only",
                                                 "TrustBFI"]
    rows = load(args.csv)

    methods = [m for m in order if any(r["policy"] == m for r in rows)]
    targets = list(args.targets)
    xpos = {t: i for i, t in enumerate(targets)}

    av_ovh = min((r["ovh"] for r in rows if r["policy"] == "Always-Verify"),
                 default=None)

    sel = {(m, t): cheapest(rows, m, t) for m in methods for t in targets}
    feasible = [v["ovh"] for v in sel.values() if v]
    ceiling = max(feasible + ([av_ovh] if av_ovh else [1.0])) * 1.8

    fig, ax = plt.subplots(figsize=(args.width, args.height))
    feasible, av = draw_panel(ax, rows, targets, methods,
                              show_ratios=not args.no_ratios)
    # A requirement the operator sets, not a measured quantity: "allowed"
    # says that, and avoids making the reader recall the symbol D_e.
    # Each point meets the target in the MEAN; individual episodes can run
    # longer.
    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)
    # av_ovh must be inside the range or the Always-Verify line is drawn off
    # the top of the axis and silently disappears
    ax.set_ylim(min(feasible) * 0.5, max(feasible + [av or 0]) * 1.8)
    # Above the axes, not inside: the Always-Verify line sits near the top of
    # the data range in every variant, so any in-axes legend collides with it.
    ps.legend_above(ax, ncol=args.legend_ncol, numpoints=1)
    ps.save(fig, args.out, args.outdir)


if __name__ == "__main__":
    main()
