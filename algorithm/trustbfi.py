# trustbfi.py
# TrustBFI -- STEP 2, file 2: the decision system. Pure logic, no dataset.
#
#   rho(A)  = rho0 * (1 - exp(-A / tau))            hazard, non-decreasing, < 1
#   q^-(t)  = q(t-1) + [1 - q(t-1)] * rho(A(t))     prediction: belief drifts up
#   q(t)    = p(s|C) q^- / [p(s|C) q^- + p(s|L)(1 - q^-)]    correction: Bayes
#   a(t)    = 1[ q(t) >= eta ],  eta = C_v / C_c
#
# WHY THE HAZARD EXISTS: it guarantees q crosses eta in bounded time even when
# the score is uninformative or adversarially forged. Without it, an attacker
# who forges benign-looking BFI silences verification completely.
#
# NUMERICS: the Beta pdf diverges at s=0 and s=1, so the raw likelihood ratio
# can reach 1e18 or 1e-18 and the belief saturates at 0 or 1 permanently. Two
# guards, both applied before any arithmetic:
#   1. s is clipped to [EPS, 1-EPS].
#   2. the per-window log-likelihood-ratio is clamped to +/- LOG_LR_CLAMP.
# The Bayes update itself runs in the log domain.
#
# HAZARD DOMINANCE. Guard 2 is not cosmetic. With an unclamped LR the belief has
# a stable fixed point at
#       q* = LR * rho0 / [ (1 - rho0) * (1 - LR) ]
# so a stream of forged s=1.0 scores pins q near zero and the hazard never wins.
# Clamping makes the robustness claim a checkable inequality:
#
#       exp(-LOG_LR_CLAMP) > 1 - rho0        ==>  q increases monotonically
#                                                 toward 1 whatever the scores,
#                                                 so it crosses any eta < 1 in
#                                                 bounded time.
#
# assert_hazard_dominates() checks it. It is a numerical robustness bound chosen
# once from that inequality.

import json
import math

# ---------------------------------------------------------------- CONFIG ----
N_F = 5                 # consecutive triggered-but-passed verifications -> key compromise
T_C = 100               # rounds without a trigger after which that counter resets (Algorithm 1)
RHO0 = 0.9              # hazard ceiling; must be < 1

# tau is DERIVED, not tuned. With the score pinned at its most reassuring value
# the hazard alone carries q across eta at t ~= 1.2*tau (measured: tau=50 fires
# at t=59). So tau follows from T_max, the longest a session may run unverified
# when nothing looks wrong:
#
#       tau = T_max / 1.2
#
# T_max = 600 windows. At W=10, S=1 and a ~20 Hz sounding rate that is roughly
# 30 s of unverified session.
#
# Setting tau far below this is what breaks unit test (c): the hazard spends its
# whole budget early, q^- sits just under eta within a few windows, and a single
# bad window then saturates the belief. At tau=50, q^-(11)=0.202 against
# eta=0.25 -- a 1.31x odds gap that one clamped window (7.39x) blows straight
# through, making an isolated dip indistinguishable from a sustained attack.
TAU = 500.0             # hazard time constant, in windows = T_max / 1.2
ETA = 0.25              # trigger threshold = C_v / C_c  (C_v = 1, C_c = 4)
EPS = 1e-6              # score clip, keeps the Beta pdf finite
LOG_LR_CLAMP = 2.0      # per-window evidence bound, in nats

# ------------------------------------------------------------------ math ----


def hazard(A, rho0=RHO0, tau=TAU):
    """rho(A) = rho0 * (1 - exp(-A/tau)). Non-decreasing in A, in [0, rho0], < 1."""
    if A <= 0:
        return 0.0
    return rho0 * (1.0 - math.exp(-A / tau))


def assert_hazard_dominates(clamp=LOG_LR_CLAMP, rho0=RHO0):
    """exp(-clamp) > 1 - rho0, the condition under which no score stream can
    hold the belief down forever."""
    lo = math.exp(-clamp)
    if not lo > 1.0 - rho0:
        raise ValueError(
            f"hazard cannot dominate: exp(-{clamp})={lo:.4f} <= 1-rho0={1 - rho0:.4f}. "
            "A forged benign score stream would pin the belief below eta forever."
        )
    return True


def _log_beta_pdf(s, a, b):
    """log Beta(s; a, b), evaluated directly so no scipy dependency is needed
    for the pure-logic layer."""
    return ((a - 1.0) * math.log(s) + (b - 1.0) * math.log1p(-s)
            - (math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)))


# --------------------------------------------------------------- tracker ----


class RiskTracker:
    """Bayesian compromise-belief tracker. q is the posterior probability that
    the session is compromised."""

    def __init__(self, likelihoods, rho0=RHO0, tau=TAU,
                 eps=EPS, log_lr_clamp=LOG_LR_CLAMP):
        self.aL = float(likelihoods["L"]["a"])
        self.bL = float(likelihoods["L"]["b"])
        self.aC = float(likelihoods["C"]["a"])
        self.bC = float(likelihoods["C"]["b"])
        self.rho0 = rho0
        self.tau = tau
        self.eps = eps
        self.clamp = log_lr_clamp
        self.q = 0.0

    @classmethod
    def from_json(cls, path="likelihoods.json", **kw):
        with open(path) as fh:
            return cls(json.load(fh), **kw)

    def reset(self):
        """q <- 0. Called only after a successful cryptographic verification."""
        self.q = 0.0

    def predict(self, A):
        """q^- = q + (1-q) * rho(A). The belief drifts up on its own with age."""
        rho = hazard(A, self.rho0, self.tau)
        self.q = self.q + (1.0 - self.q) * rho
        return self.q

    def log_lr(self, s):
        """Clamped log p(s|C) - log p(s|L), with s clipped off 0 and 1."""
        s = min(max(float(s), self.eps), 1.0 - self.eps)
        llr = _log_beta_pdf(s, self.aC, self.bC) - _log_beta_pdf(s, self.aL, self.bL)
        return max(-self.clamp, min(self.clamp, llr))

    def correct(self, s):
        """Bayes update in the log domain:
             logit q(t) = logit q^-(t) + log LR(s)
        Log-domain because p(s|C)*q^- underflows to 0 for confident scores, and
        0/0 is how you get nan beliefs."""
        q = min(max(self.q, 1e-300), 1.0 - 1e-16)
        logit = math.log(q) - math.log1p(-q)
        logit += self.log_lr(s)
        # sigmoid, written so neither branch overflows
        if logit >= 0:
            self.q = 1.0 / (1.0 + math.exp(-logit))
        else:
            e = math.exp(logit)
            self.q = e / (1.0 + e)
        return self.q


# ---------------------------------------------------------------- system ----


class TrustBFI:
    """Cost-aware trigger on top of the risk tracker."""

    name = "TrustBFI"

    def __init__(self, tracker, eta=ETA, N_f=N_F):
        self.tracker = tracker
        self.eta = eta
        self.N_f = N_f
        self.t_v = 0            # time of last successful verification
        self.c = 0              # consecutive triggered-but-passed verifications
        self.key_compromise = False

    @property
    def q(self):
        return self.tracker.q

    def step(self, s, t):
        """predict -> correct -> compare. Returns (triggered, q)."""
        self.tracker.predict(t - self.t_v)
        q = self.tracker.correct(s)
        return (q >= self.eta), q

    def gain(self, C_v=1.0, C_c=None):
        """G(t) = q*C_c - C_v > 0 is the same test as q >= eta."""
        C_c = C_c if C_c is not None else C_v / self.eta
        return self.q * C_c - C_v

    def on_verified(self, t, new_mu=None):
        """Handshake passed: the device really is the enrolled one. Belief goes
        to zero, the clock restarts, and mu is re-enrolled -- which is the ONLY
        place mu is allowed to move. Repeated triggers that keep passing mean
        the score is accusing a device the crypto keeps clearing, which points
        at a stolen key rather than a swapped device."""
        self.tracker.reset()
        self.t_v = t
        self.c += 1
        if self.c >= self.N_f:
            self.key_compromise = True
        return new_mu

    def on_failed(self):
        """Handshake failed: the transmitter is not the enrolled device."""
        return "revoke"

    def on_no_trigger(self, t=None):
        """Quiet round. The key-compromise counter resets only once the session
        has run T_C rounds since the last verification (Algorithm 1 of the
        paper). Without t, it resets on every quiet round (legacy behaviour)."""
        if t is None or t - self.t_v >= T_C:
            self.c = 0


# -------------------------------------------------------------- baselines ---
# Identical step(s, t) -> (triggered, q) signature so the harness needs no
# special-casing. q is reported as 0/1 where the baseline has no belief.


class Periodic:
    """Trigger iff A >= T_v. Ignores s entirely."""

    name = "Periodic"

    def __init__(self, T_v, N_f=N_F):
        self.T_v = T_v
        self.N_f = N_f
        self.t_v = 0
        self.c = 0
        self.key_compromise = False
        self.q = 0.0

    def step(self, s, t):
        self.q = 1.0 if (t - self.t_v) >= self.T_v else 0.0
        return self.q >= 1.0, self.q

    def on_verified(self, t, new_mu=None):
        self.t_v = t
        self.q = 0.0
        self.c += 1
        if self.c >= self.N_f:
            self.key_compromise = True
        return new_mu

    def on_failed(self):
        return "revoke"

    def on_no_trigger(self, t=None):
        """Quiet round. The key-compromise counter resets only once the session
        has run T_C rounds since the last verification (Algorithm 1 of the
        paper). Without t, it resets on every quiet round (legacy behaviour)."""
        if t is None or t - self.t_v >= T_C:
            self.c = 0


class BFIThreshold:
    """Trigger iff s < gamma. No belief, no memory, no hazard."""

    name = "BFIThreshold"

    def __init__(self, gamma, N_f=N_F):
        self.gamma = gamma
        self.N_f = N_f
        self.t_v = 0
        self.c = 0
        self.key_compromise = False
        self.q = 0.0

    def step(self, s, t):
        self.q = 1.0 if s < self.gamma else 0.0
        return self.q >= 1.0, self.q

    def on_verified(self, t, new_mu=None):
        self.t_v = t
        self.q = 0.0
        self.c += 1
        if self.c >= self.N_f:
            self.key_compromise = True
        return new_mu

    def on_failed(self):
        return "revoke"

    def on_no_trigger(self, t=None):
        """Quiet round. The key-compromise counter resets only once the session
        has run T_C rounds since the last verification (Algorithm 1 of the
        paper). Without t, it resets on every quiet round (legacy behaviour)."""
        if t is None or t - self.t_v >= T_C:
            self.c = 0


class AgeOnly:
    """Trigger iff A > A_max. Ignores s entirely."""

    name = "AgeOnly"

    def __init__(self, A_max, N_f=N_F):
        self.A_max = A_max
        self.N_f = N_f
        self.t_v = 0
        self.c = 0
        self.key_compromise = False
        self.q = 0.0

    def step(self, s, t):
        self.q = 1.0 if (t - self.t_v) > self.A_max else 0.0
        return self.q >= 1.0, self.q

    def on_verified(self, t, new_mu=None):
        self.t_v = t
        self.q = 0.0
        self.c += 1
        if self.c >= self.N_f:
            self.key_compromise = True
        return new_mu

    def on_failed(self):
        return "revoke"

    def on_no_trigger(self, t=None):
        """Quiet round. The key-compromise counter resets only once the session
        has run T_C rounds since the last verification (Algorithm 1 of the
        paper). Without t, it resets on every quiet round (legacy behaviour)."""
        if t is None or t - self.t_v >= T_C:
            self.c = 0
