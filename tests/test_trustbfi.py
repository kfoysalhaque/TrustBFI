# test_trustbfi.py
# TrustBFI -- STEP 2: the five unit tests. NO dataset required.
#
#   python -m tests.test_trustbfi        (standalone, no pytest needed)
#   pytest test_trustbfi.py        (also works if pytest is installed)
#
# The likelihoods here are synthetic and deliberately fixed:
#     p(s|L) = Beta(8, 2)   mass near 1   -- a legitimate device scores high
#     p(s|C) = Beta(2, 5)   mass low      -- a substituted device scores low
# They are a stand-in for likelihoods.json so the logic can be tested with no
# encoder, no embeddings and no scores.csv.
#
# Operating parameters come from trustbfi.py CONFIG and are NOT overridden here.

import math

from algorithm.trustbfi import (
    RiskTracker, TrustBFI, hazard, assert_hazard_dominates,
    RHO0, TAU, ETA, N_F, LOG_LR_CLAMP,
)

LIK = {"L": {"a": 8.0, "b": 2.0}, "C": {"a": 2.0, "b": 5.0}}

BENIGN = 1.0        # a legitimate window
LOW = 0.02          # a window that looks like a different device


def make_system(eta=ETA):
    return TrustBFI(RiskTracker(LIK), eta=eta, N_f=N_F)


def drive(sys_, scores, t0=1, stop_on_trigger=True):
    """Feed a score sequence. Returns (fired_at, q_at_fire, trace).
    fired_at is None if it never triggered."""
    trace = []
    for i, s in enumerate(scores):
        t = t0 + i
        triggered, q = sys_.step(s, t)
        trace.append((t, s, q, triggered))
        if triggered:
            if stop_on_trigger:
                return t, q, trace
            sys_.on_verified(t)
        else:
            sys_.on_no_trigger()
    return None, (trace[-1][2] if trace else 0.0), trace


def _fmt(trace, n=6):
    head = trace[:n]
    return "  ".join(f"t={t} s={s:.2f} q={q:.4f}" for t, s, q, _ in head)


# --------------------------------------------------------------------------
# precondition: the clamp/hazard pair must satisfy exp(-clamp) > 1 - rho0,
# otherwise tests (a) and (e) cannot pass for any run length.
# --------------------------------------------------------------------------

def test_precondition_hazard_dominates():
    assert_hazard_dominates()
    assert math.exp(-LOG_LR_CLAMP) > 1.0 - RHO0
    # hazard is non-decreasing, starts at 0, saturates strictly below 1
    assert hazard(0) == 0.0
    prev = -1.0
    for A in range(0, 2000, 10):
        h = hazard(A)
        assert h >= prev - 1e-15, f"hazard decreased at A={A}"
        assert 0.0 <= h < 1.0, f"hazard out of [0,1) at A={A}: {h}"
        prev = h
    assert hazard(10 ** 9) < 1.0
    print(f"    hazard in [0, {hazard(10**9):.4f}], non-decreasing; "
          f"exp(-{LOG_LR_CLAMP})={math.exp(-LOG_LR_CLAMP):.4f} > 1-rho0={1-RHO0:.4f}")


# --------------------------------------------------------------------------
# (a) s pinned at 1.0 -> q rises only via the hazard, and still fires in
#     bounded time.
# --------------------------------------------------------------------------

def test_a_pinned_high_still_fires():
    # Hazard-only latency is ~1.2*tau, so five time constants is a generous
    # ceiling on "bounded time" rather than a threshold tuned to the result.
    BOUND = int(5 * TAU)
    sys_ = make_system()
    fired, q, trace = drive(sys_, [BENIGN] * BOUND)

    # every window is maximally reassuring, so any rise is the hazard's doing
    assert fired is not None, (
        f"NEVER FIRED in {BOUND} windows at s=1.0 (q={q:.6g}). The hazard is "
        f"not dominating: a forged benign stream would silence verification.")
    print(f"    fired at t={fired} (q={q:.4f}) with every score pinned at 1.0")
    return fired


# --------------------------------------------------------------------------
# (b) s pinned at 0.0 -> q rises fast, fires within a few windows.
# --------------------------------------------------------------------------

def test_b_pinned_low_fires_fast():
    FEW = 10
    sys_ = make_system()
    fired, q, trace = drive(sys_, [0.0] * 50)

    assert fired is not None, f"never fired at s=0.0 (q={q:.6g})"
    assert fired <= FEW, (
        f"fired at t={fired}, expected within {FEW} windows. {_fmt(trace)}")
    print(f"    fired at t={fired} (q={q:.4f}) with every score pinned at 0.0")


# --------------------------------------------------------------------------
# (c) ONE isolated low score -> q barely moves, does NOT fire.
#     This is the entire justification for a Bayesian layer over a threshold:
#     a plain BFIThreshold(gamma=0.5) fires on this trace.
# --------------------------------------------------------------------------

def test_c_isolated_dip_does_not_fire():
    PREFIX, SUFFIX = 10, 10
    scores = [BENIGN] * PREFIX + [LOW] + [BENIGN] * SUFFIX

    sys_ = make_system()
    fired, q, trace = drive(sys_, scores)

    dip_t = PREFIX + 1
    q_dip = next(q for t, s, q, _ in trace if t == dip_t) if fired is None or fired >= dip_t else None
    assert fired is None, (
        f"FIRED at t={fired} on a single isolated dip (q={trace[fired-1][2]:.4f} "
        f"vs eta={ETA}). One bad window must not trigger a handshake.\n"
        f"      trace: {_fmt(trace, 14)}")
    print(f"    never fired over {len(scores)} windows; q at the dip = {q_dip:.4f} "
          f"< eta={ETA}; final q={q:.4f}")


# --------------------------------------------------------------------------
# (d) sustained low scores -> fires.
# --------------------------------------------------------------------------

def test_d_sustained_low_fires():
    PREFIX = 5
    scores = [BENIGN] * PREFIX + [LOW] * 40

    sys_ = make_system()
    fired, q, trace = drive(sys_, scores)

    assert fired is not None, f"never fired on sustained low scores (q={q:.6g})"
    assert fired > PREFIX, f"fired at t={fired}, before the low scores began"
    print(f"    fired at t={fired} (q={q:.4f}), "
          f"{fired - PREFIX} windows after the low scores began")


# --------------------------------------------------------------------------
# (e) forged benign: the attacker substitutes at t_a and then emits perfectly
#     benign-looking BFI forever. The score layer is blind; only the hazard
#     can save us. This is the paper's central robustness claim.
# --------------------------------------------------------------------------

def test_e_forged_benign_still_fires():
    BOUND = int(5 * TAU)        # as in (a): five hazard time constants
    sys_ = make_system()

    # a clean stretch that ends in a successful handshake, so the belief is
    # genuinely zero and the clock genuinely restarts at t_a
    t_a = 20
    for t in range(1, t_a):
        sys_.step(BENIGN, t)
        sys_.on_no_trigger()
    sys_.on_verified(t_a)
    assert sys_.q == 0.0

    # substitution happens here; every subsequent score is forged to 1.0
    fired = None
    for t in range(t_a + 1, t_a + 1 + BOUND):
        triggered, q = sys_.step(BENIGN, t)
        if triggered:
            fired = t
            break
        sys_.on_no_trigger()

    assert fired is not None, (
        f"NEVER FIRED in {BOUND} windows of forged benign score. An attacker "
        f"who forges the BFI would silence verification permanently.")
    print(f"    substitution at t={t_a}, forged s=1.0 throughout; "
          f"fired at t={fired}, {fired - t_a} windows later")


# --------------------------------------------------------------------------

TESTS = [
    ("precondition  hazard dominance", test_precondition_hazard_dominates),
    ("(a) s=1.0 pinned    -> fires in bounded time", test_a_pinned_high_still_fires),
    ("(b) s=0.0 pinned    -> fires within a few windows", test_b_pinned_low_fires_fast),
    ("(c) one isolated dip -> does NOT fire", test_c_isolated_dip_does_not_fire),
    ("(d) sustained low    -> fires", test_d_sustained_low_fires),
    ("(e) forged benign    -> still fires via hazard", test_e_forged_benign_still_fires),
]


def main():
    print("=" * 74)
    print("TrustBFI unit tests")
    print(f"  rho0={RHO0}  tau={TAU}  eta={ETA}  N_f={N_F}  clamp={LOG_LR_CLAMP}")
    print(f"  p(s|L)=Beta({LIK['L']['a']}, {LIK['L']['b']})   "
          f"p(s|C)=Beta({LIK['C']['a']}, {LIK['C']['b']})")
    print("=" * 74)

    failures = []
    for label, fn in TESTS:
        try:
            fn()
            print(f"  PASS  {label}")
        except AssertionError as e:
            failures.append((label, str(e)))
            print(f"  FAIL  {label}")
            print(f"        {e}")
        except Exception as e:  # noqa: BLE001
            failures.append((label, f"{type(e).__name__}: {e}"))
            print(f"  ERROR {label}")
            print(f"        {type(e).__name__}: {e}")

    print("=" * 74)
    n = len(TESTS)
    print(f"  {n - len(failures)}/{n} passed")
    for label, msg in failures:
        print(f"  FAILED: {label}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
