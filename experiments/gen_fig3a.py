# gen_fig3a.py
# Fig. 3 of the paper -- cost of a security level.
#
# Sweeps every policy over its own parameter and records, for each setting,
# the two quantities that matter:
#
#   exposure_mean    D_e = t_v - t_a, windows a substitution survives undetected
#   overhead_per_1k  strong verifications per 1000 windows on legitimate traffic
#
# The figure then asks: for a tolerated exposure, what does each policy COST?
# That selection lives in plots/plot_fig2.py (reused by plots/plot_fig3.py) so
# this CSV stays reusable.
#
# Two things are held identical across every policy and every setting, because
# otherwise the comparison is meaningless:
#   - the same attack sessions (victim x attacker x t_a x enrollment domain)
#   - the same post-substitution observation window (--horizon), with the
#     attacker segment truncated to it, so no policy gets longer to detect
#
# Always-Verify (a strong handshake on every sounding) is Periodic with T_v = 1.
# It is emitted under its own name because it is the security upper bound the
# paper argues is prohibitive.
#
#   python -m experiments.gen_fig3a --emb embeddings.npz --lik likelihoods.json

import os
import csv
import json
import argparse

import numpy as np

from encoder.verify import window_embeddings, score, _l2
from algorithm.trustbfi import RiskTracker, TrustBFI, Periodic, AgeOnly, N_F
from algorithm.simulate import handshake_oracle, reenroll

TRAIN_DEVICES = ["04", "0d", "0f", "10", "12", "6c", "88", "8b", "b3", "bb", "c0"]

# These policies never look at s(t): they fire purely on elapsed time. Their
# exposure is therefore identical for every victim, attacker and environment --
# visible in the raw output as every mean ending in the same fraction. Running
# them over all 90 victim x domain combinations produces 90 copies of the same
# number, so they are evaluated on a single combination instead. That buys a
# T_v grid fine enough to show the 500/D_e line without step artefacts.
SCORE_INDEPENDENT = {"Periodic", "Age-Only", "Always-Verify"}


def run_session(policy, trace, mu, enrolled, alpha, K, stop_on_revoke=True):
    n_ver, revoked = 0, None
    for t, (dev, z) in enumerate(trace):
        s = float(score(z, mu, alpha)[0])
        triggered, _ = policy.step(s, t)
        if triggered:
            n_ver += 1
            if handshake_oracle(dev, enrolled):
                new_mu = reenroll([(d, None, zz) for d, zz in trace], t, K)
                if new_mu is not None:
                    mu = new_mu
                policy.on_verified(t, new_mu)
            else:
                policy.on_failed()
                revoked = t
                if stop_on_revoke:
                    break
        else:
            policy.on_no_trigger(t)
    return n_ver, revoked


def make(kind, param, lik):
    if kind == "TrustBFI":
        return TrustBFI(RiskTracker(lik), eta=param, N_f=N_F)
    if kind in ("Periodic", "Always-Verify"):
        return Periodic(T_v=param, N_f=N_F)
    if kind == "Age-Only":
        return AgeOnly(A_max=param, N_f=N_F)
    raise ValueError(kind)


def run_tolerant(policy, trace, mu, enrolled, alpha, K):
    """Session runner for the overhead measurement when attacker windows are
    injected. A failed handshake is counted and the session CONTINUES, with the
    belief reset -- standing in for an operator who investigates and re-admits.
    The deployed behaviour is to revoke, but a revoked session ends, and then
    there is no steady state in which to measure a verification rate at all.

    Returns (n_verifications, n_detections). A verification whose handshake
    FAILS caught a real substitution -- that is value delivered, not overhead.
    Counting it as cost would penalise a policy for detecting and reward one for
    missing, so the two are reported separately."""
    n_ver = n_det = 0
    for t, (dev, z) in enumerate(trace):
        s_t = float(score(z, mu, alpha)[0])
        triggered, _ = policy.step(s_t, t)
        if triggered:
            n_ver += 1
            if handshake_oracle(dev, enrolled):
                new_mu = reenroll([(d, None, zz) for d, zz in trace], t, K)
                if new_mu is not None:
                    mu = new_mu
                policy.on_verified(t, new_mu)
            else:
                n_det += 1
                policy.on_failed()
                if hasattr(policy, "tracker"):
                    policy.tracker.reset()
                policy.t_v = t
        else:
            policy.on_no_trigger(t)
    return n_ver, n_det


def inject(trace, attacker, W_att, rate, ep_len, rng):
    """Replace `rate` of the session's windows with the attacker's own BFI, in
    contiguous episodes of `ep_len`, placed at random."""
    if rate <= 0:
        return trace
    out = list(trace)
    n_eps = max(1, int(round(rate * len(out) / ep_len)))
    for _ in range(n_eps):
        st = int(rng.integers(0, max(1, len(out) - ep_len)))
        for i in range(st, min(st + ep_len, len(out))):
            out[i] = (attacker, W_att[i % len(W_att)])
    return out


def measure(kind, param, lik, pairs, W_, victims, attackers, move_at, t_as,
            horizon, alpha, K, t_as_clock=None, attack_rate=0.0,
            ep_len=20, seed=0):
    """(overhead per 1k on legit sessions, D_e array on attack sessions, missed).

    Clock-driven policies collapse to one victim/domain (their exposure does not
    depend on either) but get MANY more substitution times, because D_e is a
    deterministic sawtooth in t_a and averaging it needs dense sampling. Score-
    driven policies need the device and environment diversity instead, so they
    keep the full victim x domain grid at a smaller t_a count. Both end up with
    a comparable number of runs; the samples are just spent where the variance
    actually is."""
    if kind in SCORE_INDEPENDENT and attack_rate <= 0:
        pairs, victims, attackers = pairs[:1], victims[:1], attackers[:1]
        t_as = t_as_clock if t_as_clock is not None else t_as
    rng = np.random.default_rng(seed)
    n_ver = n_win = n_det = 0
    for A, B in pairs:
        mus = {v: _l2(W_[A][v][:K].mean(axis=0)) for v in victims}
        for v in victims:
            for mv in move_at:
                trace = ([(v, z) for z in W_[A][v][:mv]]
                         + [(v, z) for z in W_[B][v]])
                if attack_rate > 0:
                    trace = inject(trace, attackers[0], W_[A][attackers[0]],
                                   attack_rate, ep_len, rng)
                    nv, nd = run_tolerant(make(kind, param, lik), trace,
                                          mus[v], v, alpha, K)
                    n_det += nd
                else:
                    nv, _ = run_session(make(kind, param, lik), trace, mus[v],
                                        v, alpha, K)
                n_ver += nv
                n_win += len(trace)
    overhead = 1000.0 * n_ver / max(n_win, 1)
    # verifications that caught nothing: the honest cost. Detections are value
    # delivered and are reported separately rather than charged as overhead.
    wasted = 1000.0 * (n_ver - n_det) / max(n_win, 1)
    detected = 1000.0 * n_det / max(n_win, 1)

    des, missed = [], 0
    for A, _B in pairs:
        mus = {v: _l2(W_[A][v][:K].mean(axis=0)) for v in victims}
        for v in victims:
            for a in attackers:
                for ta in t_as:
                    if ta >= len(W_[A][v]):
                        continue
                    trace = ([(v, z) for z in W_[A][v][:ta]]
                             + [(a, z) for z in W_[A][a][:horizon]])
                    _, rev = run_session(make(kind, param, lik), trace, mus[v],
                                         v, alpha, K)
                    if rev is None or (rev - ta) > horizon:
                        missed += 1
                        des.append(horizon)
                    else:
                        des.append(rev - ta)
    return overhead, np.array(des, dtype=float), missed, wasted, detected


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb", default="embeddings.npz")
    p.add_argument("--lik", default="likelihoods.json")
    p.add_argument("--out", default="results/fig3a_tradeoff.csv")
    p.add_argument("--W", type=int, default=10)
    p.add_argument("--S", type=int, default=1)
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--domains", nargs="+", type=int, default=list(range(1, 10)))
    p.add_argument("--move-at", nargs="+", type=int, default=[300, 600])
    # t_a MUST NOT be aligned with any candidate verification period. Fixed
    # values like 100,200,... put a scheduled check exactly on the substitution
    # for every T_v that divides them, handing Periodic D_e=0 for free. Drawing
    # t_a at random spreads the schedule phase uniformly, which is the whole
    # point of averaging.
    p.add_argument("--t-a", nargs="+", type=int, default=None,
                   help="explicit substitution times (default: random, seeded)")
    p.add_argument("--n-ta", type=int, default=10,
                   help="random substitution times for SCORE-DRIVEN policies")
    p.add_argument("--n-ta-clock", type=int, default=400,
                   help="random substitution times for clock-driven policies, "
                        "which are cheap and need dense t_a sampling")
    p.add_argument("--ta-range", nargs=2, type=int, default=[80, 700])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--horizon", type=int, default=1000,
                   help="post-substitution observation window, same for all policies")
    p.add_argument("--attackers", nargs="+", default=None)
    p.add_argument("--attack-rate", type=float, default=0.0,
                   help="fraction of windows in the OVERHEAD sessions that are "
                        "substituted; 0 means attack-free normal operation")
    p.add_argument("--ep-len", type=int, default=20,
                   help="length of each injected substitution episode")
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
    attackers = args.attackers or [lik["split"]["replay_attacker"]]
    doms = args.domains
    pairs = [(d, doms[(i + 1) % len(doms)]) for i, d in enumerate(doms)]
    devs = set(TRAIN_DEVICES + attackers)
    W_ = {loc: {d: window_embeddings(emb[f"{d}|{loc}"], args.W, args.S)
                for d in devs} for loc in set(doms)}

    if args.t_a:
        t_as = args.t_a
    else:
        rng = np.random.default_rng(args.seed)
        t_as = sorted(int(x) for x in rng.integers(args.ta_range[0],
                                                   args.ta_range[1], args.n_ta))
    rng2 = np.random.default_rng(args.seed + 1)
    t_as_clock = sorted(int(x) for x in rng2.integers(args.ta_range[0],
                                                      args.ta_range[1],
                                                      args.n_ta_clock))
    print(f"[fig3a] victims={victims} attackers={attackers} alpha={alpha} "
          f"horizon={args.horizon}")
    print(f"[fig3a] t_a (randomised, seed={args.seed}) = {t_as}")

    GRIDS = {
        # T_v = 1 is Always-Verify and is emitted separately below.
        # fine geometric grid: coarse steps quantise the "cheapest setting
        # meeting the target" selection and fake a staircase onto what is
        # analytically the straight line overhead = 500 / D_e
        "Periodic": ("T_v", [2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 26, 32, 40, 50,
                             64, 80, 100, 130, 160, 200, 260, 320, 400, 500,
                             640, 800, 1000, 1200]),
        "Age-Only": ("A_max", [2, 4, 8, 16, 32, 64, 130, 260, 500, 1000]),
        "TrustBFI": ("eta", [0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.45,
                             0.55, 0.65, 0.75, 0.85, 0.92, 0.98]),
        "Always-Verify": ("T_v", [1]),
    }

    rows = []
    for kind, (pname, grid) in GRIDS.items():
        for pv in grid:
            ovh, des, missed, wasted, detected = measure(
                kind, pv, lik, pairs, W_, victims,
                                       attackers, args.move_at, t_as,
                                       args.horizon, alpha, args.K,
                                       t_as_clock=t_as_clock,
                                       attack_rate=args.attack_rate,
                                       ep_len=args.ep_len, seed=args.seed)
            rows.append(dict(policy=kind, param_name=pname, param_value=pv,
                             overhead_per_1k=round(float(ovh), 4),
                             wasted_per_1k=round(float(wasted), 4),
                             detect_per_1k=round(float(detected), 4),
                             exposure_mean=round(float(des.mean()), 3),
                             exposure_median=round(float(np.median(des)), 3),
                             exposure_p95=round(float(np.percentile(des, 95)), 3),
                             missed_rate=round(missed / max(len(des), 1), 4),
                             n_attack_runs=len(des)))
            print(f"  {kind:<14} {pname}={pv:<7} overhead={ovh:8.3f}/1k  "
                  f"De={des.mean():8.2f}  missed={missed}/{len(des)}")

    fields = ["policy", "param_name", "param_value", "overhead_per_1k",
              "wasted_per_1k", "detect_per_1k",
              "exposure_mean", "exposure_median", "exposure_p95",
              "missed_rate", "n_attack_runs"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[out] {args.out}  ({len(rows)} rows)")

    # cost of a security level: cheapest setting meeting each exposure target
    print("\n[fig3a] verifications/1k needed to hold exposure at or below:")
    print(f"    {'D_e<=':>7} " + "".join(f"{k:>16}" for k in GRIDS))
    for tgt in (1, 5, 25, 100, 500):
        line = f"    {tgt:>7} "
        for kind in GRIDS:
            ok = [r for r in rows if r["policy"] == kind
                  and r["exposure_mean"] <= tgt]
            line += (f"{min(r['overhead_per_1k'] for r in ok):>16.3f}"
                     if ok else f"{'infeasible':>16}")
        print(line)


if __name__ == "__main__":
    main()
