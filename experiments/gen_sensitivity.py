# gen_sensitivity.py
# Sensitivity of the risk tracker to the hazard parameters (rho0, tau) and to
# a mismatch of the fitted likelihoods (Sec. III-B).
#
# Single-STA sessions, vectorized over many independent STAs. Scores are drawn
# from Beta densities: the tracker always uses the FITTED likelihoods of
# likelihoods.json, while the TRUE scores may come from shifted Betas (same
# concentration a+b, shifted mean) to emulate a misfit. Attacks follow
# gen_fig3a: contiguous episodes of 20 rounds covering ~5% of the rounds. A
# failed handshake ends the episode and the legitimate device is re-admitted
# (as run_tolerant). eta = 0.55 is the operating point of Fig. 3.
#
#   python -m experiments.gen_sensitivity --lik likelihoods.json

import json
import argparse

import numpy as np
from scipy import stats

from algorithm.trustbfi import RHO0, TAU, EPS, LOG_LR_CLAMP


def beta_shift(p, dmean):
    """Beta with the same concentration as p and its mean shifted by dmean."""
    c = p["a"] + p["b"]
    m = np.clip(p["a"] / c + dmean, 0.02, 0.98)
    return stats.beta(m * c, (1 - m) * c)


def run(lik, rho0, tau, dL, dC, eta, N, T, rate, ep_len, seed):
    rng = np.random.default_rng(seed)
    fL = stats.beta(lik["L"]["a"], lik["L"]["b"])          # tracker's model
    fC = stats.beta(lik["C"]["a"], lik["C"]["b"])
    tL, tC = beta_shift(lik["L"], dL), beta_shift(lik["C"], dC)   # truth
    p_start = rate / ep_len
    q = np.zeros(N); tv = np.zeros(N, dtype=int)
    left = np.zeros(N, dtype=int)                          # attack rounds left
    t0 = np.full(N, -1)
    n_ver = n_waste = 0; delays = []
    for t in range(1, T + 1):
        new = (left == 0) & (rng.random(N) < p_start)
        left[new] = ep_len; t0[new] = t
        att = left > 0
        s = np.where(att, tC.rvs(N, random_state=rng), tL.rvs(N, random_state=rng))
        s = np.clip(s, EPS, 1 - EPS)
        rho = rho0 * (1 - np.exp(-(t - tv) / tau))
        qm = np.clip(q + (1 - q) * rho, 1e-300, 1 - 1e-16)
        llr = np.clip(fC.logpdf(s) - fL.logpdf(s), -LOG_LR_CLAMP, LOG_LR_CLAMP)
        q = 1 / (1 + np.exp(-(np.log(qm) - np.log1p(-qm) + llr)))
        trig = q >= eta
        n_ver += int(trig.sum())
        n_waste += int((trig & ~att).sum())
        det = trig & att
        delays.extend((t - t0[det]).tolist())
        left[det] = 0
        q[trig] = 0.0; tv[trig] = t
        left[att & ~det] -= 1
    return 1000 * n_ver / (N * T), 1000 * n_waste / (N * T), np.mean(delays), \
        np.mean(np.array(delays) == 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lik", default="likelihoods.json")
    p.add_argument("--N", type=int, default=200)
    p.add_argument("--T", type=int, default=20000)
    p.add_argument("--eta", type=float, default=0.55)
    p.add_argument("--rate", type=float, default=0.05)
    p.add_argument("--ep-len", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    lik = json.load(open(a.lik))
    cases = [("nominal", RHO0, TAU, 0, 0),
             ("tau=250", RHO0, 250, 0, 0), ("tau=1000", RHO0, 1000, 0, 0),
             ("rho0=0.87", 0.87, TAU, 0, 0), ("rho0=0.95", 0.95, TAU, 0, 0),
             ("genuine mean -0.1", RHO0, TAU, -0.1, 0),
             ("impostor mean +0.1", RHO0, TAU, 0, +0.1),
             ("both", RHO0, TAU, -0.1, +0.1)]
    print(f"{'case':>20} | ver/1k | wasted/1k | mean D_e | D_e=0")
    for name, r0, tau, dL, dC in cases:
        ov, w, d, d0 = run(lik, r0, tau, dL, dC, a.eta, a.N, a.T, a.rate, a.ep_len, a.seed)
        print(f"{name:>20} | {ov:6.2f} | {w:9.2f} | {d:8.2f} | {d0:.3f}")


if __name__ == "__main__":
    main()
