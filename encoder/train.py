# train.py
# Stage-A training on SOURCE locations:
#   - Encoder + Projector trained with Supervised Contrastive Loss (SupCon)
#   - Optional Classifier trained with Cross-Entropy (CE) for stability
#
# Expects:
#   datagen.py  -> make_splits(...)
#   model.py    -> Encoder, Projector, Classifier
#   loss.py     -> SupConLoss
#
# Saves:
#   checkpoint.pt (encoder/projector/classifier + mac_to_y)
#
# Example:
#   python -m encoder.train /path/to/Vmatrices --source 1 2 3 4 5 6 --target 7 8 9

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from encoder.datagen import make_splits
from encoder.model import Encoder, Projector, Classifier
from encoder.loss import SupConLoss


def train_one_epoch(enc, proj, clf, loader, opt, supcon, ce_weight, device):
    enc.train()
    proj.train()
    if clf is not None:
        clf.train()

    total_loss = 0.0
    total_ce = 0.0
    total_sc = 0.0
    total_n = 0

    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)

        z = enc(x)                          # (B, emb_dim)
        u = proj(z)                         # (B, proj_dim)

        loss_sc = supcon(u, y)              # SupCon on projected space

        if clf is not None and ce_weight > 0:
            logits = clf(z)                 # CE on embedding space
            loss_ce = F.cross_entropy(logits, y)
            loss = loss_sc + ce_weight * loss_ce
        else:
            loss_ce = torch.tensor(0.0, device=device)
            loss = loss_sc

        opt.zero_grad()
        loss.backward()
        opt.step()

        b = x.size(0)
        total_loss += float(loss.item()) * b
        total_sc += float(loss_sc.item()) * b
        total_ce += float(loss_ce.item()) * b
        total_n += b

    return total_loss / total_n, total_sc / total_n, total_ce / total_n


@torch.no_grad()
def eval_ce(enc, clf, loader, device):
    enc.eval()
    clf.eval()

    correct = 0
    total = 0
    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)
        z = enc(x)
        logits = clf(z)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    return correct / max(total, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root_dir")
    p.add_argument("--source", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6])
    p.add_argument("--target", nargs="+", type=int, default=[7, 8, 9])  # only used for split creation
    p.add_argument("--k", type=int, default=5)                          # only used for split creation

    # Device holdout. Open-set verification needs impostor devices the encoder
    # has never seen, so train on a subset of mac prefixes and keep the rest out.
    p.add_argument("--devices", nargs="+", type=str, default=None,
                   help="mac prefixes to TRAIN on, e.g. 04 0d 0f ... ; default = all")
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda", "mps"],
                   help="compute device; auto = cuda if available else cpu")

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--tau", type=float, default=0.07)

    p.add_argument("--emb_dim", type=int, default=128)
    p.add_argument("--proj_dim", type=int, default=128)
    p.add_argument("--ce_weight", type=float, default=0.5)              # set 0.0 to disable CE head
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--save", type=str, default="checkpoint.pt")
    args = p.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print("[train] device:", device)
    print("[train] train devices:", args.devices if args.devices else "ALL (no holdout)")

    # Build splits (we only train on source; target splits are for later scripts)
    src_train, src_val, _, _, mac_to_y = make_splits(
        args.root_dir,
        source_locs=args.source,
        target_locs=args.target,
        k_per_class=args.k,
        val_ratio=0.2,
        seed=args.seed,
        devices=args.devices,
    )

    num_classes = len(mac_to_y)
    print("[train] num_classes:", num_classes)
    print("[train] src_train:", len(src_train), "src_val:", len(src_val))

    train_loader = DataLoader(src_train, batch_size=args.batch, shuffle=True, drop_last=True, num_workers=4)
    val_loader = DataLoader(src_val, batch_size=args.batch, shuffle=False, num_workers=4)

    # Model
    enc = Encoder(emb_dim=args.emb_dim).to(device)
    proj = Projector(emb_dim=args.emb_dim, proj_dim=args.proj_dim).to(device)

    clf = None
    if args.ce_weight > 0:
        clf = Classifier(emb_dim=args.emb_dim, num_classes=num_classes).to(device)

    # Loss + optimizer
    supcon = SupConLoss(temperature=args.tau).to(device)

    params = list(enc.parameters()) + list(proj.parameters())
    if clf is not None:
        params += list(clf.parameters())

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    best_acc = -1.0

    for ep in range(1, args.epochs + 1):
        loss, loss_sc, loss_ce = train_one_epoch(
            enc, proj, clf, train_loader, opt, supcon, args.ce_weight, device
        )

        if clf is not None:
            val_acc = eval_ce(enc, clf, val_loader, device)
        else:
            val_acc = 0.0

        print(f"[train] epoch {ep:03d} | loss={loss:.4f} (sc={loss_sc:.4f}, ce={loss_ce:.4f}) | val_acc={val_acc:.4f}")

        # Save best by val_acc (if CE enabled), else save last
        if clf is not None:
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(
                    {
                        "encoder": enc.state_dict(),
                        "projector": proj.state_dict(),
                        "classifier": clf.state_dict(),
                        "mac_to_y": mac_to_y,
                        "train_devices": sorted(mac_to_y.keys()),
                        "source_locs": args.source,
                        "emb_dim": args.emb_dim,
                        "proj_dim": args.proj_dim,
                    },
                    args.save,
                )
        else:
            # no CE head => just overwrite (or you can save every N epochs)
            torch.save(
                {
                    "encoder": enc.state_dict(),
                    "projector": proj.state_dict(),
                    "mac_to_y": mac_to_y,
                    "train_devices": sorted(mac_to_y.keys()),
                    "source_locs": args.source,
                    "emb_dim": args.emb_dim,
                    "proj_dim": args.proj_dim,
                },
                args.save,
            )

    print("[train] saved:", args.save)


if __name__ == "__main__":
    main()

#
# python -m encoder.train Dataset/vmatrices --source 1 --target 1 --save checkpoint_s1.pt
#
# TrustBFI step 1 -- 11 train devices, 4 impostors held out:
#   python -m encoder.train Dataset/vmatrices \
#       --source 1 2 3 4 5 6 --target 7 8 9 \
#       --devices 04 0d 0f 10 12 6c 88 8b b3 bb c0 \
#       --save ckpt_holdout.pt