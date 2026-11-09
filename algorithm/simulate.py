# simulate.py
# TrustBFI -- STEP 2, file 3: run the decision system on real traces.
#
# Works on embeddings.npz, NOT scores.csv, and computes s(t) ONLINE. That is
# not a stylistic choice: a successful handshake re-enrolls mu, and every score
# after that point is measured against the new mu. A fixed score column cannot
# express re-enrollment, so replaying one would silently mis-simulate the
# system.
#
# Reuses step 1 as-is: window_embeddings, enroll and score are imported from
# verify.py, never reimplemented.
#
#   python -m algorithm.simulate --emb embeddings.npz --lik likelihoods.json

import json
import argparse

import numpy as np

from encoder.verify import window_embeddings, enroll, score, _l2
from algorithm.trustbfi import RiskTracker, TrustBFI, ETA, N_F


# ------------------------------------------------------------------ traces --

def window_seq(emb, device, domain, W, S):
    """The window-embedding sequence for one (device, domain)."""
    return window_embeddings(emb[f"{device}|{domain}"], W, S)


def build_legit_trace(emb, device, domains, W, S):
    """A legitimate session walking across domains over time. The domain
    changes are what produce benign score dips -- the case that must NOT
    trigger. Returns a list of (device, domain, z)."""
    trace = []
    for dom in domains:
        for z in window_seq(emb, device, dom, W, S):
            trace.append((device, dom, z))
    return trace


def build_attack_trace(emb, victim, attacker, t_a, domains, W, S):
    """Victim windows until t_a, then the attacker's windows spliced in while
    the session keeps the victim's logical identity. The claimed identity never
    changes; only the transmitting hardware does."""
    trace = []
    for dom in domains:
        vic = window_seq(emb, victim, dom, W, S)
        att = window_seq(emb, attacker, dom, W, S)
        n = min(len(vic), len(att))
        for i in range(n):
            if len(trace) < t_a:
                trace.append((victim, dom, vic[i]))
            else:
                trace.append((attacker, dom, att[i]))
    return trace


def handshake_oracle(true_device, enrolled_device):
    """Stands in for the WPA 4-way handshake MIC check. A substituted device
    does not hold the PTK, so it cannot forge this -- success iff the device
    actually transmitting is the enrolled one."""
    return true_device == enrolled_device


# --------------------------------------------------------------------- run --

def reenroll(trace, t, K):
    """mu <- L2norm(mean of the K window embeddings starting at t).

    Same arithmetic as verify.enroll, applied at the current position instead
    of at the start of the session: enroll() always averages the FIRST K
    windows, and a mid-session re-enrollment needs the windows at t. Called
    ONLY after a successful handshake, never on the score alone."""
    chunk = trace[t:t + K]
    if not chunk:
        return None          # trigger on the very last window; keep the old mu
    zs = np.stack([z for _, _, z in chunk], axis=0)
    return _l2(zs.mean(axis=0))


def run(policy, trace, mu, enrolled_device, alpha, K):
    """Per-window loop: compute s against the CURRENT mu, step the policy, and
    on a trigger consult the oracle and dispatch. Returns the event log."""
    log = []
    for t, (dev, dom, z) in enumerate(trace):
        s = float(score(z, mu, alpha)[0])
        triggered, q = policy.step(s, t)

        oracle_result, action = "", ""
        if triggered:
            ok = handshake_oracle(dev, enrolled_device)
            oracle_result = "pass" if ok else "fail"
            if ok:
                # re-enrollment moves mu, which changes every later score
                new_mu = reenroll(trace, t, K)
                if new_mu is not None:
                    mu = new_mu
                policy.on_verified(t, new_mu)
                action = "re-enroll"
                if getattr(policy, "key_compromise", False):
                    action = "re-enroll+KEY-COMPROMISE"
            else:
                policy.on_failed()
                action = "REVOKE"
        else:
            policy.on_no_trigger(t)

        log.append({"t": t, "device": dev, "domain": dom, "s": s, "q": q,
                    "triggered": int(triggered),
                    "oracle_result": oracle_result, "action": action})

        if action == "REVOKE":
            break
    return log


def print_log(title, log, every=25, context=3):
    """Print the log: every Nth window plus every window near an event, so the
    interesting rows are never elided."""
    events = {e["t"] for e in log if e["triggered"]}
    keep = set()
    for t in events:
        keep |= set(range(t - context, t + context + 1))

    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)
    print(f"  {'t':>5} {'device':>7} {'dom':>4} {'s':>8} {'q':>8} "
          f"{'trig':>5} {'oracle':>7}  action")
    shown_gap = False
    for e in log:
        if not (e["t"] % every == 0 or e["t"] in keep or e["t"] == log[-1]["t"]):
            if not shown_gap:
                print(f"  {'...':>5}")
                shown_gap = True
            continue
        shown_gap = False
        print(f"  {e['t']:>5} {e['device']:>7} {e['domain']:>4} {e['s']:>8.4f} "
              f"{e['q']:>8.4f} {e['triggered']:>5} {e['oracle_result']:>7}  "
              f"{e['action']}")

    n_trig = sum(e["triggered"] for e in log)
    n_pass = sum(1 for e in log if e["oracle_result"] == "pass")
    n_fail = sum(1 for e in log if e["oracle_result"] == "fail")
    print(f"  -- {len(log)} windows, {n_trig} triggers "
          f"({n_pass} handshake pass, {n_fail} fail)")


# -------------------------------------------------------------------- main --

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb", default="embeddings.npz")
    p.add_argument("--lik", default="likelihoods.json")
    p.add_argument("--W", type=int, default=10)
    p.add_argument("--S", type=int, default=1)
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--alpha", type=float, default=None,
                   help="default: the alpha recorded in likelihoods.json")
    p.add_argument("--eta", type=float, default=ETA)
    p.add_argument("--enroll-dom", type=int, default=7)
    p.add_argument("--domains", nargs="+", type=int, default=[8, 9])
    p.add_argument("--t-a", type=int, default=300, help="substitution window")
    args = p.parse_args()

    emb = np.load(args.emb, allow_pickle=False)
    with open(args.lik) as fh:
        lik = json.load(fh)

    split = lik["split"]
    victim = split["replay_victim"]
    attacker = split["replay_attacker"]

    # the leakage check, enforced again at replay time
    calib = set(split["calib_genuine"]) | set(split["calib_impostor"])
    bad = sorted(calib & {victim, attacker})
    if bad:
        raise SystemExit(f"LEAKAGE: {bad} were calibrated and are being replayed")

    alpha = args.alpha if args.alpha is not None else lik.get("alpha")
    if alpha is None:
        raise SystemExit(
            "no alpha: pass --alpha, or record it with "
            "`calibrate.py --alpha <the value verify.py printed>`. It must be "
            "the same alpha that produced the scores the likelihoods were fitted on.")

    print("=" * 88)
    print("SETUP")
    print("=" * 88)
    print(f"  calibrated on   : L={split['calib_genuine']}  C={split['calib_impostor']}")
    print(f"  replayed here   : victim={victim}  attacker={attacker}  (neither calibrated)")
    print(f"  p(s|L)=Beta({lik['L']['a']:.4f}, {lik['L']['b']:.4f})   "
          f"p(s|C)=Beta({lik['C']['a']:.4f}, {lik['C']['b']:.4f})")
    print(f"  W={args.W} S={args.S} K={args.K} alpha={alpha} eta={args.eta} N_f={N_F}")
    print(f"  enroll domain={args.enroll_dom}  session domains={args.domains}")

    # Enrollment: K windows immediately after a successful strong auth, taken
    # from the enrollment domain, which is disjoint from the session domains.
    mu0 = enroll(emb, victim, args.enroll_dom, args.K, args.W, args.S)

    # ---- legit trace -----------------------------------------------------
    legit = build_legit_trace(emb, victim, args.domains, args.W, args.S)
    pol = TrustBFI(RiskTracker(lik), eta=args.eta, N_f=N_F)
    log_legit = run(pol, legit, mu0, victim, alpha, args.K)
    print_log(f"LEGIT TRACE -- device {victim} walking domains {args.domains}",
              log_legit)

    # ---- attack trace ----------------------------------------------------
    attack = build_attack_trace(emb, victim, attacker, args.t_a,
                                args.domains, args.W, args.S)
    pol = TrustBFI(RiskTracker(lik), eta=args.eta, N_f=N_F)
    log_attack = run(pol, attack, mu0, victim, alpha, args.K)
    print_log(f"ATTACK TRACE -- {victim} until t={args.t_a}, then {attacker} "
              f"substituted under {victim}'s identity", log_attack)

    fired = [e["t"] for e in log_attack if e["triggered"] and e["t"] >= args.t_a]
    if fired:
        print(f"\n  DETECTION: first trigger after substitution at t={fired[0]}, "
              f"{fired[0] - args.t_a} windows after t_a={args.t_a}")
        rev = [e["t"] for e in log_attack if e["action"] == "REVOKE"]
        print(f"  session revoked at t={rev[0]}" if rev else
              "  no revocation (oracle never failed)")
    else:
        print(f"\n  NO DETECTION after substitution at t_a={args.t_a}")


if __name__ == "__main__":
    main()
