#!/usr/bin/env bash
# Regenerate every figure at the size it is PLACED at in the paper.
#
# Printed text size = in-figure font size x (width LaTeX gives the figure /
# width it was drawn at). That RATIO is the whole problem: a 3.5 in figure in a
# 1.7 in subfigure slot renders 8 pt text at 3.9 pt, and no font bump fixes it
# while the ratio holds.
#
#   fig1  fig2  fig3   single-column floats drawn at \columnwidth -> ratio 1,
#                      so the font scale is a bump for comfort, not a
#                      correction. fig2 uses FS_FULL; fig1 and fig3 hold two
#                      small panels each and run larger (FS_FIG1, FS_FIG3).
#                      fig3 carries BOTH attack rates as two panels of one
#                      figure: one legend, one y-axis and one x-label instead
#                      of two.
#   fig4  fig5         placed as two subfigures inside ONE column, so
#                      each sits at ~0.49\columnwidth -> ratio ~0.5, and
#                      FS_HALF has to roughly double the fonts to compensate.
#                      At that size their default labels no longer fit, hence
#                      the short --ylabel, the stacked legend and --spare.
set -e
PY="${PYTHON:-python}"
FS_FULL="${FS_FULL:-1.18}"
# fig3 is two small panels in one column-wide float, so its axis text needs to
# run larger than the single-panel figures to stay comfortable at print size.
FS_FIG3="${FS_FIG3:-1.5}"
# fig1 is a two-panel trace with in-panel annotations, so it too runs larger
# than the plain single-panel figures.
FS_FIG1="${FS_FIG1:-1.45}"
FS_HALF="${FS_HALF:-1.9}"
COL=3.5      # \columnwidth
SUB=3.4      # drawn size for the half-column subfigures

# --axis-fs lifts only the ticks and axis labels; --legend-fs pulls the
# in-panel legend back down, since it sits over the trace.
$PY plot_fig1.py --width $COL --height 3.3 --fontscale "$FS_FIG1" \
    --axis-fs 1.3 --legend-fs 1.0 --legend-y 0.03
$PY plot_fig2.py --width $COL --height 2.9 --fontscale "$FS_FULL"
$PY plot_fig3.py --width $COL --height 2.3 --fontscale "$FS_FIG3" \
    --xlabel-y -0.06 --ratio-dy 5

# Legend inside the panel rather than above it: locations 3-9 all sit near
# 0.65, so the upper right is empty, and moving it there takes two text rows
# out of the figure HEIGHT -- 2.7 in down to 2.1 in.
$PY plot_fig4.py --width $SUB --height 2.1 --fontscale "$FS_HALF" \
    --ylabel "FVR (%)" --legend-inside --legend-fs 1.0
$PY plot_fig5.py --width $SUB --height 3.0 --fontscale "$FS_HALF" --spare
