# gen_multiuser.py
# Multi-user prioritization under limited reauthentication capacity (Sec. IV-D).
#
# N STAs share one AP that can run at most M handshakes per 10 rounds (token
# bucket). Scores are drawn from the fitted likelihoods in likelihoods.json:
# Beta_L while a session is legitimate, Beta_C while it is compromised. The
# belief update is trustbfi.py's (age hazard + clamped Bayes), vectorized over
# STAs. A failed handshake revokes the session and the legitimate device is
# re-admitted, so the simulation keeps running (as run_tolerant in gen_fig3a).
#
# Policies sharing the same capacity:
#   priority     TrustBFI: among STAs with q >= eta, serve the M largest
#                G_i = q_i C_c - C_v (equal costs -> largest q_i)
#   fifo         same trigger, served first-come-first-served (no priority)
#   round-robin  verify M STAs per 10 rounds in turn, ignoring q
#
#   python -m experiments.gen_multiuser --lik likelihoods.json

import json
import argparse

import numpy as np
from scipy import stats

from algorithm.trustbfi import RHO0, TAU, EPS, LOG_LR_CLAMP


def run(policy, M, lik, N, T, p_att, eta, warm, seed):
    rng = np.random.default_rng(seed)
    bL = stats.beta(lik["L"]["a"], lik["L"]["b"])
    bC = stats.beta(lik["C"]["a"], lik["C"]["b"])
    q = np.zeros(N)
    tv = np.zeros(N, dtype=int)            # last successful verification
    comp = np.zeros(N, dtype=bool)         # currently compromised
    t_att = np.full(N, -1)                 # attack onset
    cross = np.full(N, np.inf)             # round q first crossed eta (FIFO)
    tokens, rr = 0.0, 0
    delays, n_ver = [], 0
    for t in range(1, T + 1):
        new = (~comp) & (rng.random(N) < p_att)
        comp |= new
        t_att[new] = t
        s = np.where(comp, bC.rvs(N, random_state=rng), bL.rvs(N, random_state=rng))
        s = np.clip(s, EPS, 1.0 - EPS)
        # predict: q^- = q + (1-q) rho(A)
        rho = RHO0 * (1.0 - np.exp(-(t - tv) / TAU))
        qm = np.clip(q + (1.0 - q) * rho, 1e-300, 1.0 - 1e-16)
        # correct: logit q = logit q^- + clamped log LR
        llr = np.clip(bC.logpdf(s) - bL.logpdf(s), -LOG_LR_CLAMP, LOG_LR_CLAMP)
        q = 1.0 / (1.0 + np.exp(-(np.log(qm) - np.log1p(-qm) + llr)))
        trig = q >= eta
        cross[trig & np.isinf(cross)] = t
        cross[~trig] = np.inf
        # capacity: M handshakes per 10 rounds
        tokens = min(tokens + M / 10.0, float(M))
        k = int(tokens)
        if policy == "round-robin":
            sel = [(rr + j) % N for j in range(k)]
            rr = (rr + k) % N
        else:
            cand = np.flatnonzero(trig)
            key = -q[cand] if policy == "priority" else cross[cand]
            sel = cand[np.argsort(key, kind="stable")][:k]
        for i in sel:
            tokens -= 1.0
            n_ver += 1
            if comp[i]:                    # handshake fails -> detected
                if t_att[i] > warm:
                    delays.append(t - t_att[i])
                comp[i] = False            # revoked; legitimate device re-admitted
            q[i], tv[i], cross[i] = 0.0, t, np.inf
    missed = int((comp & (t_att > warm)).sum())
    return np.array(delays, dtype=float), n_ver, missed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lik", default="likelihoods.json")
    p.add_argument("--N", type=int, default=100)
    p.add_argument("--T", type=int, default=20000)
    p.add_argument("--warm", type=int, default=1000)
    p.add_argument("--p-att", type=float, default=1 / 2000,
                   help="per-round probability that a legitimate session is taken over")
    p.add_argument("--eta", type=float, default=0.55)
    p.add_argument("--Ms", nargs="+", type=int, default=[1, 2, 5, 10, 1000])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    lik = json.load(open(args.lik))

    print(f"[r1c6] N={args.N} T={args.T} p_att={args.p_att} eta={args.eta}")
    print(f"{'M/10 rounds':>12} | " + " | ".join(f"{pol:>24}" for pol in
          ["priority (TrustBFI)", "fifo", "round-robin"]))
    for M in args.Ms:
        cells = []
        for pol in ["priority", "fifo", "round-robin"]:
            d, nv, miss = run(pol, M, lik, args.N, args.T, args.p_att, args.eta,
                              args.warm, args.seed)
            cells.append(f"mean {d.mean():6.1f} p95 {np.percentile(d, 95):5.0f} m{miss}")
        print(f"{M:>12} | " + " | ".join(f"{c:>24}" for c in cells))


if __name__ == "__main__":
    main()
