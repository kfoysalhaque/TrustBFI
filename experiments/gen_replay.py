# gen_replay.py
# Perfect replay (Secs. III-B and IV-E): the adversary replays BFI that the
# legitimate device reported earlier, so its scores follow the GENUINE
# likelihood and the BFI alone cannot reveal it. Only the age hazard can force
# the next reauthentication, which the adversary then fails. D = rounds from
# the start of the replay to that reauthentication.
#
# The replay starts at age A0 after the last successful verification:
#   A0 = 0      worst case (the adversary takes over right after a verification)
#   A0 random   uniform over the legitimate verification interval
#
#   python -m experiments.gen_replay --lik likelihoods.json

import json
import argparse

import numpy as np
from scipy import stats

from algorithm.trustbfi import RHO0, TAU, EPS, LOG_LR_CLAMP


def time_to_trigger(lik, eta, n, rng, A0, Tmax=100000):
    """Rounds until q >= eta, scores drawn from Beta_L, starting at age A0
    with q = 0 (as right after a verification, then legitimate until A0)."""
    fL = stats.beta(lik["L"]["a"], lik["L"]["b"])
    fC = stats.beta(lik["C"]["a"], lik["C"]["b"])
    q = np.zeros(n); A = np.zeros(n); done = np.full(n, -1)
    for t in range(1, Tmax + 1):
        A += 1
        s = np.clip(fL.rvs(n, random_state=rng), EPS, 1 - EPS)
        rho = RHO0 * (1 - np.exp(-A / TAU))
        qm = np.clip(q + (1 - q) * rho, 1e-300, 1 - 1e-16)
        llr = np.clip(fC.logpdf(s) - fL.logpdf(s), -LOG_LR_CLAMP, LOG_LR_CLAMP)
        q = 1 / (1 + np.exp(-(np.log(qm) - np.log1p(-qm) + llr)))
        new = (q >= eta) & (done < 0)
        done[new] = t
        if (done >= 0).all():
            break
    return done


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lik", default="likelihoods.json")
    p.add_argument("--eta", type=float, default=0.55)
    p.add_argument("--n", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    lik = json.load(open(a.lik))
    rng = np.random.default_rng(a.seed)
    T = time_to_trigger(lik, a.eta, a.n, rng, 0)       # legit verification interval
    print(f"legit session, age-driven reauth interval: mean {T.mean():.0f}  "
          f"p95 {np.percentile(T, 95):.0f}  max {T.max()}")
    # replay from A0 = 0 has the same law as T (scores are genuine either way)
    print(f"replay from A0=0 (worst case): mean {T.mean():.0f}  "
          f"p95 {np.percentile(T, 95):.0f}  max {T.max()}")
    # random onset: residual time of the renewal process = size-biased
    Tb = rng.choice(T, size=a.n, p=T / T.sum())
    R = (rng.random(a.n) * Tb).astype(int)
    print(f"replay from random onset:      mean {R.mean():.0f}  "
          f"p95 {np.percentile(R, 95):.0f}  max {R.max()}")


if __name__ == "__main__":
    main()
