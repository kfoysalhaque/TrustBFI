# gen_table1.py
# Verification quality (AUC and EER, Sec. IV-A of the paper) and operating parameters.
#
# Emits results/table1.csv with four blocks:
#   representation : raw BFI cosine vs the learned embedding, at the operating point
#   W_sweep        : window length, the separation-vs-reaction-time tradeoff
#   K_sweep        : enrollment size
#   per_location   : the operating point evaluated at each location separately
#
# Enrollment and scoring are in the SAME environment throughout (the first K
# windows enroll, the remainder are scored), so nothing here measures domain
# generalization.
#
#   python -m experiments.gen_table1 --emb embeddings.npz --root Dataset/vmatrices

import os
import csv
import json
import argparse

import numpy as np

from encoder.datagen import VMatDataset
from encoder.verify import window_embeddings, auc_eer, _l2, load_encoder

TRAIN_DEVICES = ["04", "0d", "0f", "10", "12", "6c", "88", "8b", "b3", "bb", "c0"]
IMPOSTORS = ["c2", "c6", "d6", "ee"]


def raw_reports(root, device, domain):
    """The 4000-d raw BFI vector per report: exactly datagen's (16,250)
    preprocessing, flattened, with NO encoder. This is the ablation that shows
    what the learned embedding is buying."""
    ds = VMatDataset(root, locations=[domain], devices=[device],
                     normalize=True, mmap=True)
    return np.stack([ds[i][0].numpy().reshape(-1) for i in range(len(ds))])


def eval_point(get_windows, devices, impostors, W, S, K, alpha):
    """Enroll on the first K windows, score the REMAINDER of the same session.
    The enrollment windows are excluded from scoring so the genuine score is
    not measured on the very windows that defined mu."""
    gen, imp = [], []
    for dev in devices:
        zw = get_windows(dev)
        if zw.shape[0] <= K:
            continue
        mu = _l2(zw[:K].mean(axis=0))
        gen.append(np.exp(-alpha * (1.0 - zw[K:] @ mu)))
        for im in impostors:
            zi = get_windows(im)
            if zi.shape[0]:
                imp.append(np.exp(-alpha * (1.0 - zi @ mu)))
    gen = np.concatenate(gen)
    imp = np.concatenate(imp)
    a, e = auc_eer(gen, imp)
    return dict(auc=a, eer=e, s_gen=float(gen.mean()), s_imp=float(imp.mean()),
                n_gen=int(gen.size), n_imp=int(imp.size))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb", default="embeddings.npz")
    p.add_argument("--root", default="Dataset/vmatrices")
    p.add_argument("--ckpt", default="ckpt_holdout.pt")
    p.add_argument("--out", default="results/table1.csv")
    p.add_argument("--W", type=int, default=10)
    p.add_argument("--S", type=int, default=1)
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--alpha", type=float, default=5.0)
    p.add_argument("--domain", type=int, default=8, help="operating-point location")
    p.add_argument("--locations", nargs="+", type=int, default=list(range(1, 10)))
    p.add_argument("--sweep-W", nargs="+", type=int, default=[1, 5, 10, 20, 50])
    p.add_argument("--sweep-K", nargs="+", type=int, default=[1, 5, 10, 20, 50, 100])
    p.add_argument("--skip-raw", action="store_true",
                   help="skip the raw-BFI ablation (it re-reads the .npy files)")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    emb = np.load(args.emb, allow_pickle=False)
    _, ckpt_devices = load_encoder(args.ckpt)
    devices = sorted(set(TRAIN_DEVICES) & set(ckpt_devices)) or TRAIN_DEVICES

    rows = []

    def add(block, param, value, r):
        rows.append(dict(block=block, param=param, value=value, **r))
        print(f"  {block:<14} {param}={value:<6} AUC={r['auc']:.4f} "
              f"EER={r['eer']:.4f} s_gen={r['s_gen']:.3f} s_imp={r['s_imp']:.3f}")

    # ---- block 1: representation -----------------------------------------
    print("[table1] representation")
    lw = lambda d: window_embeddings(emb[f"{d}|{args.domain}"], args.W, args.S)
    add("representation", "encoder", "learned",
        eval_point(lw, devices, IMPOSTORS, args.W, args.S, args.K, args.alpha))

    if not args.skip_raw:
        cache = {}

        def rw(d):
            if d not in cache:
                cache[d] = window_embeddings(
                    raw_reports(args.root, d, args.domain), args.W, args.S)
            return cache[d]

        add("representation", "encoder", "raw_bfi",
            eval_point(rw, devices, IMPOSTORS, args.W, args.S, args.K, args.alpha))
        cache.clear()

    # ---- block 2: W sweep --------------------------------------------------
    print("[table1] W sweep")
    for W in args.sweep_W:
        f = lambda d, W=W: window_embeddings(emb[f"{d}|{args.domain}"], W, args.S)
        add("W_sweep", "W", W,
            eval_point(f, devices, IMPOSTORS, W, args.S, args.K, args.alpha))

    # ---- block 3: K sweep --------------------------------------------------
    print("[table1] K sweep")
    for K in args.sweep_K:
        add("K_sweep", "K", K,
            eval_point(lw, devices, IMPOSTORS, args.W, args.S, K, args.alpha))

    # ---- block 4: per location ---------------------------------------------
    print("[table1] per location")
    for loc in args.locations:
        f = lambda d, loc=loc: window_embeddings(emb[f"{d}|{loc}"], args.W, args.S)
        add("per_location", "location", loc,
            eval_point(f, devices, IMPOSTORS, args.W, args.S, args.K, args.alpha))

    fields = ["block", "param", "value", "auc", "eer", "s_gen", "s_imp",
              "n_gen", "n_imp"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[out] {args.out}  ({len(rows)} rows)")

    per = [r for r in rows if r["block"] == "per_location"]
    if per:
        aucs = [r["auc"] for r in per]
        print(f"[out] per-location AUC: min={min(aucs):.4f} "
              f"max={max(aucs):.4f} mean={np.mean(aucs):.4f}")


if __name__ == "__main__":
    main()
