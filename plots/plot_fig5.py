# plot_fig5.py
# Fig. 4(b) of the paper -- tolerance to corrupted feedback from a legitimate device.
# Reads results/fig3d_corruption.csv.
#
# Two stacked panels sharing one x-axis, because the point of the figure is the
# CONTRAST between them and a dual y-axis would make that unreadable:
#
#   top     the consistency score s -- the evidence, which degrades badly
#   bottom  false verification rate -- the decision, which barely moves
#
# There is no attacker: the enrolled device is transmitting throughout, so every
# verification counted is one the system did not need to spend. The dashed line
# marks the rate with clean feedback, which is the age hazard's scheduled
# re-verification and not an error; only the excess above it is caused by
# corruption.
#
#   python plot_fig5.py

import csv
import argparse

import numpy as np
import matplotlib.pyplot as plt

import plotstyle as ps


def load(path):
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            for k in ("corrupt_pct", "fv_per_1k", "fvr_pct", "s_mean", "s_p05",
                      "n_false_verif", "n_windows", "eta"):
                r[k] = float(r[k])
            rows.append(r)
    return sorted(rows, key=lambda r: r["corrupt_pct"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=ps.repo("results", "fig3d_corruption.csv"))
    p.add_argument("--out", default="fig5")
    p.add_argument("--outdir", default=None)
    p.add_argument("--width", type=float, default=3.5,
                   help="drawn width in inches. Set this to the width LaTeX "
                        "will give the figure: scaling a wide figure down is "
                        "what makes the text unreadable.")
    p.add_argument("--height", type=float, default=3.0)
    p.add_argument("--fontscale", type=float, default=1.0,
                   help="multiply every font size; for figures drawn small")
    p.add_argument("--spare", action="store_true",
                   help="drop the in-panel notes and the intermediate value "
                        "labels. At half-column scale they collide with each "
                        "other and with the traces; the caption carries them.")
    args = p.parse_args()

    ps.use(args.fontscale)
    rows = load(args.csv)
    x = np.arange(len(rows))
    # Blue rather than the usual TrustBFI magenta: this figure has a single
    # series, so there is no within-figure ambiguity, and the magenta reads
    # poorly as a thin line with a shaded fill beneath it.
    c_tb = ps.BLUE
    # The top panel traces s(t) itself, not a policy, so this is a plain colour
    # rather than a SERIES lookup -- the score is an input every method shares.
    c_sc = ps.RED

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(args.width, args.height), sharex=True,
        gridspec_kw=dict(height_ratios=[1, 1], hspace=0.16))

    # ---- top: the evidence degrades --------------------------------------
    s = [r["s_mean"] for r in rows]
    lo = [r["s_mean"] - r["s_p05"] for r in rows]
    ax1.errorbar(x, s, yerr=[lo, [0] * len(rows)], fmt="-o", color=c_sc,
                 lw=1.4, ms=ps.fs(4.5), mec="white", mew=0.7, ecolor=c_sc,
                 elinewidth=0.8, capsize=2.5, zorder=4)
    # a rotated label is limited by the PANEL height, and these panels are
    # half the figure each -- so the spare form has to be very short
    ax1.set_ylabel("Score $s$" if args.spare else "Consistency\nscore $s$")
    ax1.set_ylim(0, 1.0)
    if not args.spare:
        # redundant with the y ticks once those are large enough to read
        ax1.text(x[-1], s[-1] + 0.10, f"{s[-1]:.2f}", fontsize=ps.fs(7.5),
                 color=c_sc, ha="center", va="bottom")
        ax1.text(x[0], s[0] + 0.06, f"{s[0]:.2f}", fontsize=ps.fs(7.5),
                 color=c_sc, ha="center", va="bottom")
    if not args.spare:
        ax1.text(0.02, 0.06, "whisker: 5th percentile", transform=ax1.transAxes,
                 fontsize=ps.fs(6.5), color=ps.MUTED, ha="left", va="bottom")
    ps.tidy(ax1, grid_axis="y")

    # ---- bottom: the decision does not -----------------------------------
    # Drawn as a line, not bars. The interesting movement is 0.62 -> 0.70, and
    # a bar chart has to start at zero, which would squash that into invisible
    # slivers. A line carries no zero-baseline expectation, so the axis can be
    # scaled to the data without misleading.
    v = [r["fvr_pct"] for r in rows]          # percentage of windows
    base = v[0]
    ax2.axhline(base, color=ps.MUTED, lw=0.9, ls="--", zorder=2)
    ax2.fill_between(x, base, v, color=c_tb, alpha=0.18, zorder=3)
    ax2.plot(x, v, "-o", color=c_tb, lw=1.6, ms=ps.fs(4.5), mec="white",
             mew=0.7, zorder=4)
    # only the endpoints when spare: the middle labels are the ones that touch
    labelled = [] if args.spare else range(len(rows))
    for xi, r in zip(x, rows):
        if xi not in labelled:
            continue
        ax2.text(xi, r["fvr_pct"] + (max(v) - base) * 0.10,
                 f"{r['fvr_pct']:.2f}", fontsize=ps.fs(7.5), color=ps.INK,
                 ha="center", va="bottom", zorder=6)
    # shaded band = the excess actually caused by corruption
    if not args.spare:
        ax2.annotate(f"+{max(v) - base:.2f} pts\n(+{100 * (max(v) - base) / base:.0f}% more\nhandshakes)",
                     xy=(x[-1], (base + max(v)) / 2), xytext=(-6, 0),
                     textcoords="offset points", fontsize=ps.fs(7.0),
                     color=ps.INK, ha="right", va="center", linespacing=1.2,
                     zorder=6)

    ax2.set_ylabel("FVR (%)" if args.spare
                   else "False verification\nrate (%)")
    span = max(v) - base
    ax2.set_ylim(base - span * 0.45, max(v) + span * 0.55)
    ax2.set_xlabel("Corrupted BFI (%)" if args.spare
                   else "Corrupted BFI (% of report entries)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{r['corrupt_pct']:g}" for r in rows])
    ps.tidy(ax2, grid_axis="y")
    ax2.grid(axis="x", visible=False)

    ps.save(fig, args.out, args.outdir)


if __name__ == "__main__":
    main()
