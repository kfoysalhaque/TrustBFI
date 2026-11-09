# verify.py
# TrustBFI -- STEP 1: continual verification from BFI.
#
# Sliding window over BFI reports -> window embedding -> score against the
# ENROLLED reference mu_i for the claimed device.
#
#   z_i(t) = L2norm( mean over the W reports in the window of g_theta(report) )
#   mu_i   = L2norm( (1/K) sum_{k=1..K} z_i(k) )      # K windows right after strong auth
#   d_i(t) = cosine_distance( z_i(t), mu_i )
#   s_i(t) = exp( -alpha * d_i(t) )                   # in [0,1]
#
# ANCHORING: s_i(t) always compares the current window to the ENROLLMENT
# reference mu_i. Never to the previous X windows. Scoring against recent
# history is change-point detection: after a device substitution the attacker's
# own BFI becomes the recent history, the score returns to normal, and the
# compromise goes invisible. mu_i is re-enrolled only after a successful
# cryptographic verification -- that is step 3, not here.
#
# The `rolling_mean` column in scores.csv is DIAGNOSTIC OUTPUT ONLY. It is never
# fed back into s(t). Step 2 consumes the raw s(t).
#
# The encoder takes ONE report of shape (16,250): each report is embedded
# separately and W embeddings are mean-pooled afterwards.

import os
import csv
import glob
import argparse

import numpy as np
import torch

from encoder.datagen import VMatDataset, parse_filename
from encoder.model import Encoder


# ----------------------------------------------------------------------------
# encoder
# ----------------------------------------------------------------------------

def load_encoder(ckpt_path, device="cpu"):
    """Load Encoder from model.py, eval mode, frozen. No gradients anywhere."""
    # weights_only=False: our own checkpoint carries mac_to_y / train_devices,
    # which are plain python objects, not tensors.
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    emb_dim = ck.get("emb_dim", 128)

    enc = Encoder(emb_dim=emb_dim).to(device)
    enc.load_state_dict(ck["encoder"])
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)

    # train.py records exactly which devices the encoder saw. mac_to_y is the
    # fallback for checkpoints written before that key existed.
    train_devices = ck.get("train_devices") or sorted(ck.get("mac_to_y", {}).keys())
    return enc, sorted(train_devices)


# ----------------------------------------------------------------------------
# per-report embeddings (the expensive step -- cached)
# ----------------------------------------------------------------------------

def embed_reports(root, out, ckpt, batch=512, device="cpu", num_workers=0):
    """Embed every report in the dataset once, cache to `out` keyed by
    '<device>|<domain>'. Reuses VMatDataset so the (16,250) preprocessing chain
    is matched exactly rather than reimplemented."""
    if os.path.exists(out):
        print(f"[embed] cache hit: {out}")
        return np.load(out, allow_pickle=False)

    enc, _ = load_encoder(ckpt, device=device)

    files = sorted(glob.glob(os.path.join(root, "*.npy")))
    cache = {}

    for f in files:
        mac, loc = parse_filename(f)

        # One dataset per (device, domain) file so report order is preserved:
        # a sliding window over time requires the original packet order.
        ds = VMatDataset(root, locations=[loc], devices=[mac], normalize=True, mmap=True)
        # num_workers=0: one DataLoader per file x 135 files, so worker
        # spawn cost would dominate the (cheap) mmap reads.
        loader = torch.utils.data.DataLoader(
            ds, batch_size=batch, shuffle=False, num_workers=num_workers
        )

        zs = []
        with torch.no_grad():
            for x, _, _ in loader:
                zs.append(enc(x.to(device)).cpu().numpy())

        z = np.concatenate(zs, axis=0).astype(np.float32)   # (N, emb_dim)
        cache[f"{mac}|{loc}"] = z
        print(f"[embed] {mac}_{loc}: {z.shape}")

    np.savez_compressed(out, **cache)
    print(f"[embed] wrote {out} ({len(cache)} device-domain pairs)")
    return np.load(out, allow_pickle=False)


# ----------------------------------------------------------------------------
# windows / enrollment / score
# ----------------------------------------------------------------------------

def dataset_devices(root):
    """Every mac prefix present in the dataset directory."""
    return sorted({parse_filename(f)[0]
                   for f in glob.glob(os.path.join(root, "*.npy"))})


def _l2(v, axis=-1):
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + 1e-8)


def window_embeddings(z_reports, W, S):
    """Sliding window of W consecutive report embeddings, stride S.
    Mean-pool each window, then L2-normalize. Returns (n_windows, emb_dim)."""
    n = z_reports.shape[0]
    if n < W:
        return np.zeros((0, z_reports.shape[1]), dtype=np.float32)

    starts = range(0, n - W + 1, S)
    z = np.stack([z_reports[s:s + W].mean(axis=0) for s in starts], axis=0)
    return _l2(z).astype(np.float32)


def enroll(emb, device, domain, K, W, S):
    """mu = mean of the FIRST K window embeddings, L2-normalized.
    These are the K windows collected immediately after a successful strong
    authentication."""
    zw = window_embeddings(emb[f"{device}|{domain}"], W, S)
    if zw.shape[0] < K:
        raise ValueError(f"{device}|{domain}: only {zw.shape[0]} windows, need K={K}")
    return _l2(zw[:K].mean(axis=0))


def score(z, mu, alpha):
    """s = exp(-alpha * cosine_distance(z, mu)). z may be (D,) or (n,D)."""
    z = np.atleast_2d(z)
    d = 1.0 - _l2(z) @ _l2(mu)          # cosine distance in [0,2]
    return np.exp(-alpha * d)


def pick_alpha(genuine_d, impostor_d,
               candidates=(0.5, 1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 75, 100, 150, 200)):
    """Choose alpha so genuine scores concentrate near 1 and impostor scores
    spread toward 0. Objective: maximise mean(s_genuine) - mean(s_impostor).

    alpha is a monotone rescaling of the distance, so it cannot change AUC or
    EER at all -- it only sets how the scores are spread across [0,1]. It is
    chosen on distances alone and is deliberately NOT tuned to make the gate
    pass."""
    best, best_gap = None, -np.inf
    print("[alpha] sweep (AUC is alpha-invariant; this only shapes the [0,1] spread)")
    for a in candidates:
        g = np.exp(-a * genuine_d).mean()
        i = np.exp(-a * impostor_d).mean()
        print(f"[alpha]   alpha={a:>4}  mean(s_gen)={g:.4f}  mean(s_imp)={i:.4f}  gap={g - i:.4f}")
        if g - i > best_gap:
            best, best_gap = a, g - i
    print(f"[alpha] chosen alpha={best} (gap={best_gap:.4f})")
    if best in (candidates[0], candidates[-1]):
        print("[alpha] WARNING: argmax sits on the edge of the grid -- the gap "
              "is monotone here, which usually means the two distance "
              "distributions barely differ. Check GATE 1 before trusting this.")
    return best


def score_trace(emb, device, domains, mu, alpha, W, S, roll=10):
    """Score a whole session: one device walking across domains over time.
    Returns (s, rolling_mean, domain_of_each_window). The rolling mean is
    diagnostic output for inspecting the trend, NOT an input to the score."""
    s_all, dom_all = [], []
    for dom in domains:
        zw = window_embeddings(emb[f"{device}|{dom}"], W, S)
        if zw.shape[0] == 0:
            continue
        # every window scored against the ENROLLED mu, never against history
        s_all.append(score(zw, mu, alpha))
        dom_all += [dom] * zw.shape[0]

    s = np.concatenate(s_all) if s_all else np.zeros(0)

    # causal rolling mean over the last `roll` scores (diagnostic column only)
    rm = np.empty_like(s)
    for t in range(s.shape[0]):
        rm[t] = s[max(0, t - roll + 1):t + 1].mean()

    return s, rm, dom_all


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------

def auc_eer(genuine, impostor):
    """AUC (genuine scored higher than impostor) and EER, from raw scores."""
    from sklearn.metrics import roc_curve, auc as _auc

    y = np.r_[np.ones_like(genuine), np.zeros_like(impostor)]
    s = np.r_[genuine, impostor]
    fpr, tpr, _ = roc_curve(y, s)
    a = _auc(fpr, tpr)

    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[i] + fnr[i]) / 2.0
    return a, eer


def gate_scores(emb, enroll_dom, score_doms, train_devices, impostor_devices,
                alpha, W, S, K):
    """Genuine  = enrolled device, DIFFERENT domain than enrollment.
    Impostor = held-out device, same scoring domains, against that same mu.
    Returns (genuine_scores, impostor_scores, genuine_d, impostor_d)."""
    gen_d, imp_d = [], []

    for dev in train_devices:
        try:
            mu = enroll(emb, dev, enroll_dom, K, W, S)
        except (KeyError, ValueError):
            continue

        for dom in score_doms:
            zw = window_embeddings(emb[f"{dev}|{dom}"], W, S)
            if zw.shape[0]:
                gen_d.append(1.0 - zw @ mu)

            for imp in impostor_devices:
                zi = window_embeddings(emb[f"{imp}|{dom}"], W, S)
                if zi.shape[0]:
                    imp_d.append(1.0 - zi @ mu)

    gen_d = np.concatenate(gen_d)
    imp_d = np.concatenate(imp_d)
    return np.exp(-alpha * gen_d), np.exp(-alpha * imp_d), gen_d, imp_d


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="ckpt_holdout.pt")
    p.add_argument("--root", default="Dataset/vmatrices")
    p.add_argument("--W", type=int, default=10)
    p.add_argument("--S", type=int, default=1)
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--alpha", type=float, default=None, help="default: pick_alpha")
    p.add_argument("--out", default="scores.csv")
    p.add_argument("--emb", default="embeddings.npz")
    p.add_argument("--enroll-dom", type=int, default=7)
    p.add_argument("--score-doms", nargs="+", type=int, default=[8, 9])
    p.add_argument("--roll", type=int, default=10)
    p.add_argument("--sweep-W", nargs="+", type=int, default=[1, 5, 10, 20, 50])
    p.add_argument("--sweep-K", nargs="+", type=int, default=[1, 5, 10, 20, 50, 100])
    p.add_argument("--train-devices", nargs="+", default=None,
                   help="default: read from the checkpoint")
    p.add_argument("--impostor-devices", nargs="+", default=None,
                   help="default: every dataset device the encoder never saw")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, ckpt_train_devices = load_encoder(args.ckpt, device=device)
    emb = embed_reports(args.root, args.emb, args.ckpt, device=device)

    # The device split is derived from the checkpoint, not hardcoded, so the
    # holdout claim below is a fact about the encoder rather than a promise.
    all_devices = dataset_devices(args.root)
    train_devices = sorted(args.train_devices or ckpt_train_devices)
    impostor_devices = sorted(args.impostor_devices
                              or (set(all_devices) - set(train_devices)))
    if not impostor_devices:
        raise SystemExit("no impostor devices: the encoder saw every device in "
                         "the dataset, so open-set verification is untestable.")

    # ---- prove the holdout is real ---------------------------------------
    print("\n" + "=" * 74)
    print("SETUP")
    print("=" * 74)
    print(f"  devices in dataset         : {len(all_devices)} {all_devices}")
    print(f"  encoder trained on devices : {len(train_devices)} {train_devices}")
    print(f"  impostor devices           : {len(impostor_devices)} {impostor_devices}")
    leak = sorted(set(impostor_devices) & set(ckpt_train_devices))
    print(f"  impostors held out of training: {not leak}" + (f"  LEAK: {leak}" if leak else ""))
    print(f"  enrollment domain          : {args.enroll_dom}")
    print(f"  scoring domains            : {args.score_doms}")
    ov = sorted(set([args.enroll_dom]) & set(args.score_doms))
    print(f"  enroll/score domains disjoint : {not ov}" + (f"  OVERLAP: {ov}" if ov else ""))
    print(f"  W={args.W}  S={args.S}  K={args.K}")

    # ---- alpha ------------------------------------------------------------
    print("\n" + "=" * 74)
    print("ALPHA")
    print("=" * 74)
    _, _, gd, idd = gate_scores(emb, args.enroll_dom, args.score_doms,
                                train_devices, impostor_devices, 1.0,
                                args.W, args.S, args.K)
    alpha = args.alpha if args.alpha is not None else pick_alpha(gd, idd)

    # ---- gate 1: genuine vs impostor --------------------------------------
    gen = np.exp(-alpha * gd)
    imp = np.exp(-alpha * idd)
    a, e = auc_eer(gen, imp)

    print("\n" + "=" * 74)
    print(f"GATE 1 -- genuine vs impostor  (W={args.W}, S={args.S}, K={args.K}, alpha={alpha})")
    print("=" * 74)
    print(f"  genuine  n={gen.size:>7}  s mean={gen.mean():.4f}  std={gen.std():.4f}  "
          f"cos-dist mean={gd.mean():.4f}")
    print(f"  impostor n={imp.size:>7}  s mean={imp.mean():.4f}  std={imp.std():.4f}  "
          f"cos-dist mean={idd.mean():.4f}")
    print(f"  AUC={a:.4f}   EER={e:.4f}")

    # ---- gate 2: W sweep ---------------------------------------------------
    print("\n" + "=" * 74)
    print("GATE 2 -- W sweep (longer window = less noise, slower reaction)")
    print("=" * 74)
    print(f"  {'W':>4} {'AUC':>8} {'EER':>8} {'s_gen':>8} {'s_imp':>8} {'reports to react':>18}")
    for w in args.sweep_W:
        g_, i_, gdw, idw = gate_scores(emb, args.enroll_dom, args.score_doms,
                                       train_devices, impostor_devices, alpha,
                                       w, args.S, args.K)
        aw, ew = auc_eer(g_, i_)
        print(f"  {w:>4} {aw:>8.4f} {ew:>8.4f} {g_.mean():>8.4f} {i_.mean():>8.4f} {w:>18}")

    # ---- gate 3: K sweep ---------------------------------------------------
    print("\n" + "=" * 74)
    print("GATE 3 -- K sweep (how many enrollment windows are actually needed)")
    print("=" * 74)
    print(f"  {'K':>4} {'AUC':>8} {'EER':>8} {'s_gen':>8} {'s_imp':>8}")
    for k in args.sweep_K:
        try:
            g_, i_, _, _ = gate_scores(emb, args.enroll_dom, args.score_doms,
                                       train_devices, impostor_devices, alpha,
                                       args.W, args.S, k)
        except ValueError as ex:
            print(f"  {k:>4}  skipped: {ex}")
            continue
        ak, ek = auc_eer(g_, i_)
        print(f"  {k:>4} {ak:>8.4f} {ek:>8.4f} {g_.mean():>8.4f} {i_.mean():>8.4f}")

    # ---- score traces ------------------------------------------------------
    # Session: the enrolled device walks across the scoring domains over time,
    # then each impostor is substituted in and scored against the SAME mu.
    enrolled = train_devices[0]
    mu = enroll(emb, enrolled, args.enroll_dom, args.K, args.W, args.S)

    rows = []
    for dev in [enrolled] + impostor_devices:
        s, rm, doms = score_trace(emb, dev, args.score_doms, mu, alpha,
                                  args.W, args.S, roll=args.roll)
        for t, (sv, rv, dv) in enumerate(zip(s, rm, doms)):
            rows.append([t, dev, dv, f"{sv:.6f}", f"{rv:.6f}",
                         int(dev == enrolled)])

    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "device", "domain", "s", "rolling_mean", "is_enrolled_device"])
        w.writerows(rows)

    print(f"\n[out] {args.out}  ({len(rows)} rows, enrolled device = {enrolled})")
    print(f"[out] {args.emb}")


if __name__ == "__main__":
    main()
