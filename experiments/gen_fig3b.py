# gen_fig3b.py
# Fig. 4(a) of the paper -- false verification rate under benign domain variation.
#
# The failure mode this isolates is channel variation mistaken for compromise.
# So the experiment is: enroll in environment A, then let the LEGITIMATE device
# move to environment B, and count verifications the move provokes.
#
#   same  : enrolled at A, still at A (no move)
#   cross : enrolled at A, now at B != A -- benign mobility
#
# Sessions run through gen_fig3a.run_session, the SAME simulator fig3a and
# fig3d use. That matters for two reasons:
#
#   1. a passing handshake re-enrolls mu, so the score recovers after the move
#      instead of staying depressed for the rest of the session. Scoring every
#      window against a frozen enrollment mean is not what the system does, and
#      it inflates the rate for any stateful policy by a large factor.
#   2. the numbers it produces are the same quantity fig3a and fig3d report --
#      verifications per window on attack-free traffic -- so the three figures
#      can be read against each other.
#
# Each method's operating point is calibrated ONCE, in the enrollment
# environment, to a common VERIFICATION BUDGET, and then applied unchanged
# everywhere else, because at deployment you cannot retune per location. A
# method that only looks good after per-location retuning is not deployable.
#
#   Raw BFI          threshold on cosine similarity of the raw (16,250) report
#   Score threshold  threshold on s(t) -- the encoder, no belief, no memory
#   TrustBFI         the full belief filter
#
# See calibrate() for why the match is on cost rather than on detection: on
# this data every setting of every method detects the substitution immediately,
# so a detection target binds nothing. Detection is still MEASURED at the
# chosen points, per substitution episode exactly as fig3a measures it, and the
# run aborts if the two methods do not land within --max-delay-gap of each
# other's speed.
#
#   python -m experiments.gen_fig3b --emb embeddings.npz --lik likelihoods.json
#
# Runtime is a few minutes: the operating point cannot be read off a
# score histogram, because a stateful policy's cost and detection depend on the
# whole session, so each candidate has to be simulated.

import os
import csv
import json
import argparse

import numpy as np

from encoder.datagen import VMatDataset
from encoder.verify import window_embeddings, score, _l2
from algorithm.trustbfi import RiskTracker, TrustBFI, BFIThreshold, N_F, RHO0
from experiments.gen_fig3a import run_session, TRAIN_DEVICES


def raw_windows(root, device, domain, W, S):
    ds = VMatDataset(root, locations=[domain], devices=[device],
                     normalize=True, mmap=True)
    reports = np.stack([ds[i][0].numpy().reshape(-1) for i in range(len(ds))])
    return window_embeddings(reports, W, S)


def session_trace(W_A, W_B, v, move_at):
    """Legitimate session: `move_at` windows at A, then the device moves to B.
    For the same-environment control there is no move and the session is just
    the device's own windows at A."""
    if W_B is W_A:
        return [(v, z) for z in W_A[v]]
    return [(v, z) for z in W_A[v][:move_at]] + [(v, z) for z in W_B[v]]


def fvr(factory, W_A, W_B, victims, K, alpha, move_at):
    """Verifications per window on attack-free sessions enrolled at A.

    There is no attacker anywhere in these sessions, so every verification the
    policy triggers is a false one. Returns (rate, n_verifications, n_windows).
    """
    n_ver = n_win = 0
    for v in victims:
        if len(W_A[v]) <= K:
            continue
        mu = _l2(W_A[v][:K].mean(axis=0))
        for mv in (move_at if W_B is not W_A else [None]):
            trace = session_trace(W_A, W_B, v, mv)
            nv, _ = run_session(factory(), trace, mu, v, alpha, K)
            n_ver += nv
            n_win += len(trace)
    return n_ver / max(n_win, 1), n_ver, n_win


def delay(factory, W_A, victims, attackers, K, alpha, t_as, horizon):
    """Mean detection delay D_e = t_v - t_a over substitution episodes.

    Same construction and the same statistic as gen_fig3a.measure(): the victim
    transmits at A until t_a, then the attacker's own BFI takes over and we
    watch for `horizon` windows. A miss is charged the full horizon, so a
    policy cannot buy a short mean by simply never firing.

    Returns (mean D_e, fraction of episodes detected at all)."""
    des, hit = [], 0
    for v in victims:
        if len(W_A[v]) <= K:
            continue
        mu = _l2(W_A[v][:K].mean(axis=0))
        for a in attackers:
            for ta in t_as:
                if ta >= len(W_A[v]):
                    continue
                trace = ([(v, z) for z in W_A[v][:ta]]
                         + [(a, z) for z in W_A[a][:horizon]])
                _, rev = run_session(factory(), trace, mu, v, alpha, K)
                if rev is None or (rev - ta) > horizon:
                    des.append(float(horizon))
                else:
                    des.append(float(rev - ta))
                    hit += 1
    return float(np.mean(des)), hit / max(len(des), 1)


def calibrate(name, make, grid, W_A, victims, attackers, K, alpha,
              t_as, horizon, budget, move_at):
    """Operating point whose verification cost in the ENROLLMENT environment is
    closest to `budget` (verifications per 1000 windows).

    Why a budget and not a detection target. On this data the attacker is so
    separable at the enrollment location that BOTH methods detect the
    substitution in its first window at essentially every setting on the grid:
    the detection rate is 1.000 and the mean delay is 0 almost everywhere. A
    constraint that every candidate satisfies does not match anything -- it
    just selects whichever end of the grid is cheapest, which for a threshold
    is the 0.1st percentile of the score distribution.

    Equal cost at home is the constraint that actually binds here, and it makes
    the figure's claim precise: given the same handshake budget in the
    enrollment environment, which method wastes fewer handshakes once the
    device moves? The budget is set at A only -- no cross-location result is
    consulted -- so this is not an oracle choice.

    The measured D_e is carried through so main() can check that the two
    budget-matched points really do detect at comparable speed -- the premise
    is verified rather than assumed."""
    scored = []
    for p in grid:
        rate, _, _ = fvr(lambda: make(p), W_A, W_A, victims, K, alpha, move_at)
        scored.append((abs(1000 * rate - budget), float(p), rate))
        print(f"    {name} param={p:<8.4g} fvr@A={1000*rate:7.3f}/1k")
    _, param, rate = min(scored)
    de, det = delay(lambda: make(param), W_A, victims, attackers, K, alpha,
                    t_as, horizon)
    print(f"  [{name}] operating point = {param:.4g}  "
          f"(cost at A {1000*rate:.2f}/1k vs budget {budget:.2f}, "
          f"D_e {de:.2f}, detected {det:.3f})")
    return param, de, det


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb", default="embeddings.npz")
    p.add_argument("--lik", default="likelihoods.json")
    p.add_argument("--root", default="Dataset/vmatrices")
    p.add_argument("--out", default="results/fig3b_fvr.csv")
    p.add_argument("--W", type=int, default=10)
    p.add_argument("--S", type=int, default=1)
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--rho0", type=float, default=RHO0)
    p.add_argument("--locations", nargs="+", type=int, default=list(range(1, 10)))
    p.add_argument("--enroll-loc", type=int, default=8,
                   help="environment the operating point is calibrated in")
    p.add_argument("--budget-per-1k", type=float, default=6.0,
                   help="verifications per 1000 windows every method is "
                        "calibrated to IN THE ENROLLMENT ENVIRONMENT. The "
                        "default sits at fig3d's clean-feedback operating "
                        "point (6.16/1k), so the figures share a cost scale.")
    p.add_argument("--max-delay-gap", type=float, default=5.0,
                   help="how far apart the two budget-matched points' mean "
                        "detection delays may be before the run aborts as "
                        "non-comparable. Guards the case where a method looks "
                        "cheap only because it is slow.")
    p.add_argument("--horizon", type=int, default=100,
                   help="windows observed after t_a; a miss is charged this "
                        "much delay. Same for every method.")
    p.add_argument("--move-at", nargs="+", type=int, default=[300, 600],
                   help="window at which the device moves A -> B")
    p.add_argument("--n-ta", type=int, default=8)
    p.add_argument("--ta-range", nargs=2, type=int, default=[80, 700])
    p.add_argument("--n-grid", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-raw", action="store_true")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    emb = np.load(args.emb, allow_pickle=False)
    with open(args.lik) as fh:
        lik = json.load(fh)
    alpha = args.alpha if args.alpha is not None else lik.get("alpha")
    if alpha is None:
        raise SystemExit("no alpha in likelihoods.json; pass --alpha")

    calib = set(lik["split"]["calib_genuine"]) | set(lik["split"]["calib_impostor"])
    victims = [d for d in TRAIN_DEVICES if d not in calib]
    attackers = [lik["split"]["replay_attacker"]]
    devs = sorted(set(victims + attackers))
    A = args.enroll_loc

    rng = np.random.default_rng(args.seed)
    t_as = sorted(rng.integers(args.ta_range[0], args.ta_range[1],
                               size=args.n_ta).tolist())
    print(f"[fig3b] victims={victims} attackers={attackers} enroll_loc={A}")
    print(f"[fig3b] budget={args.budget_per_1k}/1k  "
          f"max_delay_gap={args.max_delay_gap}  horizon={args.horizon}  "
          f"t_a={t_as}")

    def learned(loc):
        return {d: window_embeddings(emb[f"{d}|{loc}"], args.W, args.S) for d in devs}

    def raw(loc):
        return {d: raw_windows(args.root, d, loc, args.W, args.S) for d in devs}

    reps = [("Score threshold", learned)] + ([] if args.skip_raw else [("Raw BFI", raw)])

    rows = []
    for rep_name, getter in reps:
        wa = getter(A)

        # gamma grid: quantiles of the genuine score distribution at A, which
        # is where a threshold's useful range lives
        s_gen = np.concatenate([
            np.exp(-alpha * (1.0 - wa[v][args.K:] @ _l2(wa[v][:args.K].mean(axis=0))))
            for v in victims if len(wa[v]) > args.K])
        g_grid = np.unique(np.quantile(
            s_gen, np.linspace(0.001, 0.60, args.n_grid)))

        print(f"  calibrating {rep_name} (threshold)...")
        gamma, de_g, det_g = calibrate(
            rep_name, lambda g: BFIThreshold(gamma=g, N_f=N_F), g_grid,
            wa, victims, attackers, args.K, alpha, t_as, args.horizon,
            args.budget_per_1k, args.move_at)

        eta = de_e = det_e = None
        if rep_name == "Score threshold":    # TrustBFI rides on the same s(t)
            e_grid = np.geomspace(0.01, 0.95, args.n_grid)
            print("  calibrating TrustBFI (belief)...")
            eta, de_e, det_e = calibrate(
                "TrustBFI",
                lambda e: TrustBFI(RiskTracker(lik, rho0=args.rho0),
                                   eta=e, N_f=N_F),
                e_grid, wa, victims, attackers, args.K, alpha, t_as,
                args.horizon, args.budget_per_1k, args.move_at)

            # the comparability premise, checked rather than asserted in prose
            if abs(de_e - de_g) > args.max_delay_gap:
                raise SystemExit(
                    f"[fig3b] budget-matched points detect at very different "
                    f"speeds: {rep_name} D_e={de_g:.2f} vs TrustBFI "
                    f"D_e={de_e:.2f} (gap > {args.max_delay_gap}). The FVR "
                    f"bars would not be comparable.")

        for B in args.locations:
            wb = wa if B == A else getter(B)
            cond = "same" if B == A else "cross"

            r, nv, nw = fvr(lambda: BFIThreshold(gamma=gamma, N_f=N_F),
                            wa, wb, victims, args.K, alpha, args.move_at)
            rows.append(dict(method=rep_name, enroll_loc=A, score_loc=B,
                             condition=cond, fvr_pct=round(100 * r, 4),
                             fv_per_1k=round(1000 * r, 4),
                             detection=round(det_g, 4), de_mean=round(de_g, 3),
                             operating_param=round(gamma, 4),
                             n_false_verif=nv, n_windows=nw))
            print(f"  [{rep_name:<14}] loc {B}: {100*r:6.3f}%  ({nv}/{nw})")

            if eta is not None:
                r, nv, nw = fvr(
                    lambda: TrustBFI(RiskTracker(lik, rho0=args.rho0),
                                     eta=eta, N_f=N_F),
                    wa, wb, victims, args.K, alpha, args.move_at)
                rows.append(dict(method="TrustBFI", enroll_loc=A, score_loc=B,
                                 condition=cond, fvr_pct=round(100 * r, 4),
                                 fv_per_1k=round(1000 * r, 4),
                                 detection=round(det_e, 4),
                                 de_mean=round(de_e, 3),
                                 operating_param=round(eta, 4),
                                 n_false_verif=nv, n_windows=nw))
                print(f"  [{'TrustBFI':<14}] loc {B}: {100*r:6.3f}%  ({nv}/{nw})")
            if B != A:
                del wb

    fields = ["method", "enroll_loc", "score_loc", "condition", "fvr_pct",
              "fv_per_1k", "detection", "de_mean", "operating_param",
              "n_false_verif", "n_windows"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[out] {args.out}  ({len(rows)} rows)")

    for m in sorted({r["method"] for r in rows}):
        for c in ("same", "cross"):
            v = [r["fvr_pct"] for r in rows if r["method"] == m and r["condition"] == c]
            if v:
                print(f"[fig3b] {m:<14} {c:<6} FVR mean={np.mean(v):7.3f}%  "
                      f"max={max(v):7.3f}%")


if __name__ == "__main__":
    main()
