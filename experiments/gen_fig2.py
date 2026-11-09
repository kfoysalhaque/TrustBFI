# gen_fig2.py
# Fig. 2 of the paper -- mechanism walkthrough on one session.
#
# The trace contains every event the paper needs to show at once:
#   * a BENIGN environment change, which dips the score. The belief filter
#     should absorb it (or trigger once and pass the handshake) -- a bare BFI
#     threshold acts on it.
#   * one or more DEVICE SUBSTITUTION episodes. The attacker transmits under
#     the victim's identity for a stretch of windows, then backs off. Each
#     onset should be caught within a window or two.
#
# --on-fail controls what happens when the handshake fails:
#   revoke          the deployed behaviour: session killed, trace ends
#   log-and-reset   ILLUSTRATION ONLY: the detection is recorded and the belief
#                   reset, as if an operator had investigated and re-admitted
#                   the session, so a single figure can show several episodes
#                   being caught instead of just the first
#
# The environment change is benign MOBILITY. The encoder trained on every
# location, so this is not a domain-generalisation test.
#
#   python -m experiments.gen_fig2      # defaults reproduce Fig. 2 of the paper

import os
import csv
import json
import argparse

import numpy as np

from encoder.verify import window_embeddings, enroll, score, _l2
from algorithm.trustbfi import RiskTracker, TrustBFI, ETA, N_F
from algorithm.simulate import handshake_oracle, reenroll


def parse_episodes(specs):
    """['320:60', '520:40'] -> [(320, 380), (520, 560)]"""
    out = []
    for sp in specs or []:
        a, b = sp.split(":")
        out.append((int(a), int(a) + int(b)))
    return sorted(out)


def build_trace(emb, victim, attacker, enroll_dom, move_dom, t_move, episodes,
                W, S, n_windows):
    """Victim throughout, in enroll_dom then move_dom, with attacker windows
    spliced in during each episode. The claimed identity never changes; only
    the transmitting hardware does."""
    v0 = window_embeddings(emb[f"{victim}|{enroll_dom}"], W, S)
    v1 = window_embeddings(emb[f"{victim}|{move_dom}"], W, S)
    a1 = window_embeddings(emb[f"{attacker}|{move_dom}"], W, S)
    a0 = window_embeddings(emb[f"{attacker}|{enroll_dom}"], W, S)

    trace = []
    for t in range(n_windows):
        moved = t >= t_move
        attacking = any(s <= t < e for s, e in episodes)
        if attacking:
            src, dev, dom = (a1 if moved else a0), attacker, (move_dom if moved else enroll_dom)
        else:
            src, dev, dom = (v1 if moved else v0), victim, (move_dom if moved else enroll_dom)
        if t >= len(src):
            break
        trace.append((dev, dom, src[t], attacking))
    return trace


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb", default="embeddings.npz")
    p.add_argument("--lik", default="likelihoods.json")
    p.add_argument("--out", default="results/fig2_trace.csv")
    p.add_argument("--W", type=int, default=10)
    p.add_argument("--S", type=int, default=1)
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--eta", type=float, default=ETA)
    p.add_argument("--enroll-dom", type=int, default=8)
    p.add_argument("--move-dom", type=int, default=9)
    p.add_argument("--t-move", type=int, default=150)
    p.add_argument("--episodes", nargs="+", default=["320:50", "500:35", "660:60"],
                   help="substitution episodes as start:length")
    p.add_argument("--n-windows", type=int, default=850)
    p.add_argument("--on-fail", choices=["revoke", "log-and-reset"],
                   default="log-and-reset")
    p.add_argument("--gamma", type=float, default=0.5,
                   help="BFI-threshold counterfactual")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    emb = np.load(args.emb, allow_pickle=False)
    with open(args.lik) as fh:
        lik = json.load(fh)

    victim = lik["split"]["replay_victim"]
    attacker = lik["split"]["replay_attacker"]
    alpha = args.alpha if args.alpha is not None else lik.get("alpha")
    if alpha is None:
        raise SystemExit("no alpha in likelihoods.json; pass --alpha")

    episodes = parse_episodes(args.episodes)
    trace = build_trace(emb, victim, attacker, args.enroll_dom, args.move_dom,
                        args.t_move, episodes, args.W, args.S, args.n_windows)
    plain = [(d, m, z) for d, m, z, _ in trace]

    mu = enroll(emb, victim, args.enroll_dom, args.K, args.W, args.S)
    pol = TrustBFI(RiskTracker(lik), eta=args.eta, N_f=N_F)

    rows, detections = [], []
    for t, (dev, dom, z, attacking) in enumerate(trace):
        s = float(score(z, mu, alpha)[0])
        triggered, q = pol.step(s, t)

        oracle_result, action = "", ""
        if triggered:
            ok = handshake_oracle(dev, victim)
            oracle_result = "pass" if ok else "fail"
            if ok:
                new_mu = reenroll(plain, t, args.K)
                if new_mu is not None:
                    mu = new_mu
                pol.on_verified(t, new_mu)
                action = "re-enroll"
            else:
                pol.on_failed()
                detections.append(t)
                if args.on_fail == "revoke":
                    action = "REVOKE"
                else:
                    # illustration: operator investigated, session re-admitted
                    action = "DETECT"
                    pol.tracker.reset()
                    pol.t_v = t
        else:
            pol.on_no_trigger(t)

        rows.append(dict(t=t, device=dev, domain=dom,
                         attacker_active=int(attacking),
                         s=round(s, 6), q=round(q, 6), eta=args.eta,
                         triggered=int(triggered),
                         bfi_threshold_would_fire=int(s < args.gamma),
                         oracle_result=oracle_result, action=action))
        if action == "REVOKE":
            break

    fields = ["t", "device", "domain", "attacker_active", "s", "q", "eta",
              "triggered", "bfi_threshold_would_fire", "oracle_result", "action"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # per-episode detection latency
    lat = []
    for (s0, e0) in episodes:
        d = next((t for t in detections if s0 <= t < e0), None)
        lat.append(None if d is None else d - s0)

    meta = dict(victim=victim, attacker=attacker, alpha=alpha, eta=args.eta,
                W=args.W, S=args.S, K=args.K, gamma=args.gamma,
                enroll_dom=args.enroll_dom, move_dom=args.move_dom,
                t_move=args.t_move, episodes=episodes, detections=detections,
                latencies=lat, on_fail=args.on_fail, n_windows=len(rows))
    with open(args.out.replace(".csv", "_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    benign = [r for r in rows if not r["attacker_active"]]
    fp_bfi = sum(r["bfi_threshold_would_fire"] for r in benign)
    fp_tb = sum(1 for r in benign if r["action"] == "re-enroll")
    print(f"[fig2] victim={victim} attacker={attacker} alpha={alpha} eta={args.eta}")
    print(f"[fig2] episodes={episodes}")
    print(f"[fig2] detected at {detections}  latency per episode = {lat} windows")
    print(f"[fig2] on benign windows: TrustBFI verified {fp_tb}x, "
          f"BFI-threshold(gamma={args.gamma}) would have fired {fp_bfi}x")
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
