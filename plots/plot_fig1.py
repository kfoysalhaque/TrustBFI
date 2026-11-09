# plot_fig1.py
# Fig. 2 of the paper -- mechanism walkthrough. Reads results/fig2_trace.csv.
#
# Two stacked panels on one time axis:
#   top     s(t), the consistency score, with the BFI-threshold line and every
#           window that threshold would have fired on
#   bottom  q(t), the compromise belief, with the trigger threshold eta
#
# Shaded bands mark the windows where the attacker is actually transmitting.
# A marker sits at each substitution onset and at the window TrustBFI first
# flags it, so the detection latency for every episode is readable directly.
#
#   python plot_fig1.py

import csv
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt

import plotstyle as ps

ATTACK_BAND = "#fbe3da"     # tint of the Periodic/orange slot


def load(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ("t", "triggered", "bfi_threshold_would_fire", "attacker_active"):
            r[k] = int(r[k])
        for k in ("s", "q", "eta"):
            r[k] = float(r[k])
    return rows


def spans(rows, key):
    """Contiguous [start, end) runs where row[key] is truthy."""
    out, start = [], None
    for r in rows:
        if r[key] and start is None:
            start = r["t"]
        elif not r[key] and start is not None:
            out.append((start, r["t"]))
            start = None
    if start is not None:
        out.append((start, rows[-1]["t"] + 1))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=ps.repo("results", "fig2_trace.csv"))
    p.add_argument("--meta", default=ps.repo("results", "fig2_trace_meta.json"))
    p.add_argument("--out", default="fig1")
    p.add_argument("--outdir", default=None)
    p.add_argument("--width", type=float, default=3.5,
                   help="drawn width in inches. Set this to the width LaTeX "
                        "will give the figure: scaling a wide figure down is "
                        "what makes the text unreadable.")
    p.add_argument("--height", type=float, default=2.4)
    p.add_argument("--fontscale", type=float, default=1.0,
                   help="multiply every font size; for figures drawn small")
    p.add_argument("--axis-fs", type=float, default=1.0,
                   help="extra multiplier on tick labels and axis labels only")
    p.add_argument("--legend-y", type=float, default=0.08,
                   help="legend height inside the s(t) panel, as an axes "
                        "fraction measured to its lower-left corner")
    p.add_argument("--legend-fs", type=float, default=1.0,
                   help="legend size, as a fraction of the base. The legend "
                        "sits INSIDE the trace here, so it competes with the "
                        "data for room in a way the axis text does not.")
    args = p.parse_args()

    ps.use(args.fontscale, axis_scale=args.axis_fs)
    rows = load(args.csv)
    try:
        with open(args.meta) as fh:
            meta = json.load(fh)
    except FileNotFoundError:
        meta = {}

    t = [r["t"] for r in rows]
    s = [r["s"] for r in rows]
    q = [r["q"] for r in rows]
    eta = rows[0]["eta"]
    gamma = meta.get("gamma", 0.5)
    t_move = meta.get("t_move")
    episodes = spans(rows, "attacker_active")

    # first flag inside each episode = that episode's detection
    det = {}
    for (a, b) in episodes:
        hit = next((r["t"] for r in rows
                    if a <= r["t"] < b and r["oracle_result"] == "fail"), None)
        if hit is not None:
            det[(a, b)] = hit

    c_tb = ps.MAGENTA_DARK          # thin traces need the deeper tone
    # The baseline here thresholds s(t), not the raw report -- "Score
    # threshold", the same policy fig5 plots. It keeps the blue slot rather
    # than the red one because red is the event accent in this figure.
    c_bfi, _, _ = ps.style("Score threshold")

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(args.width, args.height), sharex=True,
        gridspec_kw=dict(height_ratios=[1, 1], hspace=0.14))

    for ax in (ax1, ax2):
        for (a, b) in episodes:
            ax.axvspan(a, b, color=ATTACK_BAND, zorder=0, lw=0)
        if t_move is not None:
            ax.axvline(t_move, color=ps.MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)

    # ---- top: consistency score -------------------------------------------
    fp = [r["t"] for r in rows
          if r["bfi_threshold_would_fire"] and not r["attacker_active"]]
    if fp:
        ax1.plot(fp, [rows[x]["s"] for x in fp], linestyle="none", marker="v",
                 ms=2.2, color=c_bfi, alpha=0.75, zorder=3,
                 label=f"Threshold false fires ({len(fp)})")
    ax1.plot(t, s, color=c_tb, lw=1.0, zorder=4)
    ax1.axhline(gamma, color=c_bfi, lw=0.9, ls=":", zorder=2)
    ax1.text(t[-1], gamma, r"$\gamma$ ", color=c_bfi, fontsize=ps.fs(8.5),
             va="bottom", ha="right")
    ax1.set_ylabel("$s(t)$")
    ax1.set_ylim(0, 1.05)
    # Lifted clear of the axis and given a white backing. It sits over the
    # trace region, and without a background both the small marker glyph and
    # the text disappear into the plotted line.
    ax1.legend(loc="lower left", bbox_to_anchor=(0.012, args.legend_y),
               handlelength=1.2, borderpad=0.4, handletextpad=0.5,
               fontsize=ps.fs(8) * args.legend_fs,
               frameon=True, facecolor="white",
               edgecolor="none", framealpha=0.94).set_zorder(9)
    ps.tidy(ax1, grid_axis="y")

    # ---- bottom: compromise belief -----------------------------------------
    ax2.plot(t, q, color=c_tb, lw=1.1, zorder=4)
    ax2.axhline(eta, color=ps.INK, lw=0.9, ls="--", zorder=2)
    ax2.text(t[-1], eta, r"$\eta$ ", color=ps.INK, fontsize=ps.fs(8.5),
             va="bottom", ha="right")
    for (a, b), d in det.items():
        ax2.plot([d], [rows[d]["q"]], marker="*", ms=9, color=ps.ORANGE,
                 mec="white", mew=0.6, zorder=7)
    ax2.set_ylabel("$q(t)$")
    ax2.set_ylim(0, 1.05)
    # Stride S=1, so each index is one new BFI report -- one comm. round.
    # The scoring window spans W=10 rounds, but it ADVANCES one at a time.
    ax2.set_xlabel("Comm. round $t$")
    ps.tidy(ax2, grid_axis="y")

    # ---- event annotations --------------------------------------------------
    for i, (a, b) in enumerate(episodes):
        ax1.annotate("", xy=(a, 1.02), xytext=(a, 1.18),
                     arrowprops=dict(arrowstyle="-|>", color=ps.ORANGE, lw=1.0),
                     annotation_clip=False)
        lat = det.get((a, b), a) - a
        # "#2"/"#3" rather than "sub #2"/"sub #3": episodes 2 and 3 are only
        # 160 rounds apart, and the longer tags overlap at any font size worth
        # reading. The caption says what the numbers mark.
        ax1.text(a, 1.20, f"#{i+1}\n$D_e$={lat}", fontsize=ps.fs(7.5),
                 color=ps.ORANGE, ha="center", va="bottom", linespacing=1.15,
                 clip_on=False)
    if t_move is not None:
        ax1.text(t_move, 1.20, "env.\nchange", fontsize=ps.fs(7.5), color=ps.MUTED,
                 ha="center", va="bottom", linespacing=1.15, clip_on=False)

    ax2.set_xlim(min(t), max(t))
    ps.save(fig, args.out, args.outdir)


if __name__ == "__main__":
    main()
