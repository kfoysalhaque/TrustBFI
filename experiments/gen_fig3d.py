# gen_fig3d.py
# Fig. 4(b) of the paper -- tolerance to CORRUPTED feedback from a legitimate device.
#
# No attacker anywhere. The enrolled device keeps transmitting, but a fraction
# of its BFI is corrupted -- interference, quantisation damage, a lossy or
# tampered feedback channel. Every verification the system triggers is
# therefore a FALSE one, and the question is how much corruption it absorbs
# before it starts spending handshakes it did not need to.
#
# Corruption is applied to the RAW (16,250) report and the report is then
# re-embedded, because the encoder is non-linear -- perturbing the embedding
# directly would measure something else. A fraction `c` of the report's entries
# is replaced with noise matched to that report's own scale, and the report is
# renormalised afterwards so it still satisfies datagen's unit-norm convention.
#
# Enrollment happens on CLEAN reports (the device authenticated in good
# conditions) and corruption begins afterwards, which is the realistic ordering
# and the only one where the score can actually move.
#
# Sessions run through the same simulator as fig3a, so a passing handshake
# re-enrolls mu and the score recovers -- without that the score stays
# depressed forever and the false-alarm rate is wildly overstated.
#
#   python -m experiments.gen_fig3d --ckpt ckpt_holdout.pt --lik likelihoods.json

import os
import csv
import json
import argparse

import numpy as np
import torch

from encoder.datagen import VMatDataset
from encoder.verify import load_encoder, window_embeddings, _l2
from algorithm.trustbfi import RiskTracker, TrustBFI, N_F
from experiments.gen_fig3a import run_session, TRAIN_DEVICES


def raw_reports(root, device, loc):
    """Reports exactly as datagen prepares them: (N, 16, 250), unit-norm."""
    ds = VMatDataset(root, locations=[loc], devices=[device],
                     normalize=True, mmap=True)
    return np.stack([ds[i][0].numpy() for i in range(len(ds))])


def corrupt(reports, frac, rng, start):
    """Replace `frac` of each report's entries with noise, from `start` on.

    Noise is scaled to that report's own standard deviation so the corruption
    is relative rather than a fixed absolute perturbation, and each report is
    renormalised to unit norm to match how the encoder was trained."""
    out = reports.copy()
    if frac <= 0:
        return out
    for i in range(start, len(out)):
        x = out[i]
        mask = rng.random(x.shape) < frac
        if mask.any():
            x = x.copy()
            x[mask] = rng.normal(0.0, float(x.std()), int(mask.sum()))
            out[i] = x / (np.linalg.norm(x) + 1e-8)
    return out


@torch.no_grad()
def embed(enc, reports, batch=512):
    zs = []
    for i in range(0, len(reports), batch):
        xb = torch.from_numpy(reports[i:i + batch]).float()
        zs.append(enc(xb).cpu().numpy())
    return np.concatenate(zs, axis=0).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="ckpt_holdout.pt")
    p.add_argument("--root", default="Dataset/vmatrices")
    p.add_argument("--lik", default="likelihoods.json")
    p.add_argument("--out", default="results/fig3d_corruption.csv")
    p.add_argument("--W", type=int, default=10)
    p.add_argument("--S", type=int, default=1)
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--loc", type=int, default=8)
    p.add_argument("--fracs", nargs="+", type=float,
                   default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                   help="fraction of each report's entries corrupted")
    p.add_argument("--eta", type=float, default=0.04642,
                   help="operating point; default matches fig3c")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.lik) as fh:
        lik = json.load(fh)
    alpha = args.alpha if args.alpha is not None else lik.get("alpha")
    if alpha is None:
        raise SystemExit("no alpha in likelihoods.json; pass --alpha")

    calib = set(lik["split"]["calib_genuine"]) | set(lik["split"]["calib_impostor"])
    victims = [d for d in TRAIN_DEVICES if d not in calib]
    enc, _ = load_encoder(args.ckpt)

    # enrollment reports stay clean; corruption starts after them
    start = args.K + args.W
    print(f"[fig3d] location={args.loc}  devices={victims}")
    print(f"[fig3d] eta={args.eta}  clean enrollment prefix = {start} reports")

    clean = {v: raw_reports(args.root, v, args.loc) for v in victims}

    rows = []
    for frac in args.fracs:
        rng = np.random.default_rng(args.seed)
        n_ver = n_win = 0
        s_vals = []
        for v in victims:
            reps = corrupt(clean[v], frac, rng, start)
            zw = window_embeddings(embed(enc, reps), args.W, args.S)
            mu = _l2(zw[:args.K].mean(axis=0))
            s_vals.append(np.exp(-alpha * (1.0 - zw[args.K:] @ mu)))

            trace = [(v, z) for z in zw]
            pol = TrustBFI(RiskTracker(lik), eta=args.eta, N_f=N_F)
            nv, _ = run_session(pol, trace, mu, v, alpha, args.K)
            n_ver += nv
            n_win += len(trace)

        s_all = np.concatenate(s_vals)
        rows.append(dict(corrupt_frac=frac, corrupt_pct=round(100 * frac, 1),
                         eta=args.eta,
                         fv_per_1k=round(1000.0 * n_ver / max(n_win, 1), 4),
                         fvr_pct=round(100.0 * n_ver / max(n_win, 1), 4),
                         n_false_verif=n_ver, n_windows=n_win,
                         s_mean=round(float(s_all.mean()), 4),
                         s_p05=round(float(np.percentile(s_all, 5)), 4)))
        print(f"  corrupt={100*frac:5.1f}%  false verif={n_ver:4d} "
              f"({1000.0*n_ver/max(n_win,1):6.2f}/1k)  "
              f"s_mean={s_all.mean():.3f}  s_p05={np.percentile(s_all,5):.3f}")

    fields = ["corrupt_frac", "corrupt_pct", "eta", "fv_per_1k", "fvr_pct",
              "n_false_verif", "n_windows", "s_mean", "s_p05"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[out] {args.out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
