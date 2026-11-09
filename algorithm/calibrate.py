# calibrate.py
# TrustBFI -- STEP 2, file 1: fit the score likelihoods. Run once, offline.
#
#   p(s | L) = Beta(a_L, b_L)     L = legitimate
#   p(s | C) = Beta(a_C, b_C)     C = compromised
#
# LEAKAGE: the scores fitted here MUST be disjoint from the ones simulate.py
# replays. The split is by DEVICE and is written into likelihoods.json so it can
# be audited after the fact, and simulate.py re-checks it at run time.
#
#   python -m algorithm.calibrate --scores scores.csv --out likelihoods.json

import csv
import json
import argparse

import numpy as np
from scipy import stats

EPS = 1e-6                      # Beta pdf diverges at 0 and 1; clip before fitting
LR_GRID = (0.1, 0.3, 0.5, 0.7, 0.9)

# Device split. Anything in CALIB_* is fitted here and must never be replayed;
# anything in REPLAY_* is reserved for simulate.py and never fitted.
CALIB_GENUINE = ["04"]                  # p(s|L)
CALIB_IMPOSTOR = ["c2", "c6", "d6"]     # p(s|C)   -- note: 'ee' deliberately absent
REPLAY_VICTIM = "c0"                    # train device, absent from scores.csv
REPLAY_ATTACKER = "ee"                  # impostor held back from calibration


def load_scores(path):
    """Read scores.csv from step 1 -> list of dicts with typed fields."""
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "t": int(r["t"]),
                "device": r["device"],
                "domain": int(r["domain"]),
                "s": float(r["s"]),
                "is_enrolled_device": int(r["is_enrolled_device"]),
            })
    return rows


def split_calibration(rows, calib_genuine=CALIB_GENUINE,
                      calib_impostor=CALIB_IMPOSTOR,
                      replay=(REPLAY_VICTIM, REPLAY_ATTACKER)):
    """Split by device. Returns (genuine_scores, impostor_scores, split_record).

    Raises if a calibration device is also a replay device -- that would leak
    the replayed distribution into the likelihoods and make every downstream
    belief optimistic."""
    calib = set(calib_genuine) | set(calib_impostor)
    overlap = sorted(calib & set(replay))
    if overlap:
        raise SystemExit(f"LEAKAGE: device(s) {overlap} are both calibrated and replayed")

    gen = np.array([r["s"] for r in rows if r["device"] in set(calib_genuine)])
    imp = np.array([r["s"] for r in rows if r["device"] in set(calib_impostor)])

    present = sorted({r["device"] for r in rows})
    missing = sorted(calib - set(present))
    if missing:
        raise SystemExit(f"calibration devices absent from scores.csv: {missing}")

    split = {
        "by": "device",
        "calib_genuine": sorted(calib_genuine),
        "calib_impostor": sorted(calib_impostor),
        "replay_victim": replay[0],
        "replay_attacker": replay[1],
        "held_out_of_calibration": sorted(set(present) - calib) + [replay[0]],
        "n_genuine": int(gen.size),
        "n_impostor": int(imp.size),
    }
    return gen, imp, split


def fit_beta(scores):
    """MLE fit on [0,1]. Scores are clipped off the open interval's endpoints
    first, because the Beta likelihood is unbounded there."""
    x = np.clip(np.asarray(scores, dtype=float), EPS, 1.0 - EPS)
    a, b, loc, scale = stats.beta.fit(x, floc=0, fscale=1)
    return float(a), float(b)


def goodness_of_fit(name, scores, a, b, nbins=10):
    """KS statistic plus an empirical-vs-fitted histogram, printed."""
    x = np.clip(np.asarray(scores, dtype=float), EPS, 1.0 - EPS)
    ks = stats.kstest(x, "beta", args=(a, b))

    print(f"\n  [{name}]  Beta(a={a:.4f}, b={b:.4f})   n={x.size}")
    print(f"    empirical mean={x.mean():.4f} std={x.std():.4f}   "
          f"fitted mean={a / (a + b):.4f}")
    print(f"    KS D={ks.statistic:.4f}  p={ks.pvalue:.3e}"
          + ("   <-- POOR FIT" if ks.statistic > 0.15 else ""))

    edges = np.linspace(0, 1, nbins + 1)
    emp, _ = np.histogram(x, bins=edges, density=True)
    fit = np.array([stats.beta.cdf(hi, a, b) - stats.beta.cdf(lo, a, b)
                    for lo, hi in zip(edges[:-1], edges[1:])]) * nbins
    print(f"    {'bin':>12} {'empirical':>10} {'fitted':>10}")
    for lo, hi, e, f in zip(edges[:-1], edges[1:], emp, fit):
        print(f"    [{lo:.1f},{hi:.1f}) {e:>10.3f} {f:>10.3f}")
    return float(ks.statistic)


def likelihood_ratio_table(aL, bL, aC, bC, grid=LR_GRID):
    """p(s|C)/p(s|L) on the grid. MUST be monotonically decreasing in s: a high
    score has to be evidence FOR legitimacy, or the belief update runs backwards."""
    print("\n  likelihood ratio  p(s|C) / p(s|L)")
    print(f"    {'s':>6} {'p(s|L)':>12} {'p(s|C)':>12} {'LR':>14}")
    lrs = []
    for s in grid:
        pL = float(stats.beta.pdf(s, aL, bL))
        pC = float(stats.beta.pdf(s, aC, bC))
        lr = pC / pL if pL > 0 else np.inf
        lrs.append(lr)
        print(f"    {s:>6.1f} {pL:>12.5g} {pC:>12.5g} {lr:>14.5g}")

    decreasing = all(x > y for x, y in zip(lrs, lrs[1:]))
    print(f"\n  monotonically decreasing in s: {decreasing}")
    return lrs, decreasing


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", default="scores.csv")
    p.add_argument("--out", default="likelihoods.json")
    # Recorded, not used here. simulate.py must score with the SAME alpha that
    # produced scores.csv, or the fitted likelihoods describe a different
    # quantity than the one being thresholded.
    p.add_argument("--alpha", type=float, required=True,
                   help="the alpha verify.py used to write scores.csv")
    args = p.parse_args()

    rows = load_scores(args.scores)
    gen, imp, split = split_calibration(rows)

    print("=" * 74)
    print("SPLIT  (calibration devices are disjoint from replay devices)")
    print("=" * 74)
    for k, v in split.items():
        print(f"  {k:<26} {v}")

    print("\n" + "=" * 74)
    print("BETA FITS")
    print("=" * 74)
    aL, bL = fit_beta(gen)
    aC, bC = fit_beta(imp)
    ks_L = goodness_of_fit("L  legitimate", gen, aL, bL)
    ks_C = goodness_of_fit("C  compromised", imp, aC, bC)

    print("\n" + "=" * 74)
    print("LIKELIHOOD RATIO TABLE")
    print("=" * 74)
    lrs, decreasing = likelihood_ratio_table(aL, bL, aC, bC)

    if not decreasing:
        raise SystemExit(
            "\nSTOP: the likelihood ratio is not monotonically decreasing in s.\n"
            "The calibration is wrong and the belief update would move in the\n"
            "wrong direction. Not writing likelihoods.json."
        )

    out = {
        "L": {"a": aL, "b": bL},
        "C": {"a": aC, "b": bC},
        "split": split,
        "alpha": args.alpha,
        "gof": {"ks_L": ks_L, "ks_C": ks_C},
        "lr_table": {str(s): lr for s, lr in zip(LR_GRID, lrs)},
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()
