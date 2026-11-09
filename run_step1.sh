#!/usr/bin/env bash
# TrustBFI -- STEP 1: continual verification from BFI.
# Run this on the training machine. Two stages: train the encoder, then verify.
#
#   ./run_step1.sh          # both stages
#   ./run_step1.sh train    # encoder only
#   ./run_step1.sh verify   # scoring only (needs ckpt_holdout.pt)
#
# Prereqs: torch, numpy, scikit-learn, and Dataset/vmatrices/*.npy
# (135 files, 4.6 GB, from https://huggingface.co/datasets/foysalhaque/BeamID)

set -euo pipefail

PY="${PY:-python}"
ROOT="${ROOT:-Dataset/vmatrices}"
CKPT="${CKPT:-ckpt_holdout.pt}"

# 11 train devices. The other 4 (c2 c6 d6 ee) are impostors and are held out of
# encoder training entirely -- open-set verification is meaningless otherwise.
TRAIN_DEVICES="04 0d 0f 10 12 6c 88 8b b3 bb c0"

# Source locs 1-6 train the encoder. Locs 7-9 are unseen: 7 enrolls, 8-9 score,
# so the enrollment domain is disjoint from the scoring domains.
SOURCE="1 2 3 4 5 6"
TARGET="7 8 9"
ENROLL_DOM=7
SCORE_DOMS="8 9"

# W=window length in reports, S=stride, K=enrollment windows.
W="${W:-10}"
S="${S:-1}"
K="${K:-20}"

# On Apple silicon, DEVICE=mps is roughly 3-5x faster than cpu.
DEVICE="${DEVICE:-auto}"
EPOCHS="${EPOCHS:-30}"

stage="${1:-all}"

if [ "$stage" = "all" ] || [ "$stage" = "train" ]; then
  echo "=== [1/2] training encoder (holdout: $TRAIN_DEVICES) ==="
  $PY -m encoder.train "$ROOT" \
    --source $SOURCE \
    --target $TARGET \
    --devices $TRAIN_DEVICES \
    --device "$DEVICE" \
    --epochs "$EPOCHS" \
    --save "$CKPT"
fi

if [ "$stage" = "all" ] || [ "$stage" = "verify" ]; then
  echo "=== [2/2] verification + sanity gate ==="
  # verify.py reads the train/impostor split back out of the checkpoint.
  $PY -m encoder.verify \
    --ckpt "$CKPT" \
    --root "$ROOT" \
    --W "$W" --S "$S" --K "$K" \
    --enroll-dom "$ENROLL_DOM" \
    --score-doms $SCORE_DOMS \
    --emb embeddings.npz \
    --out scores.csv
fi
