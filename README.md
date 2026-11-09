# TrustBFI: Cost-Aware Continuous Verification for Zero-Trust Wi-Fi Using MIMO Beamforming Feedback

This repository contains the code and results of the paper

> K. F. Haque and R. Zabin, "TrustBFI: Cost-Aware Continuous Verification for
> Zero-Trust Wi-Fi Using MIMO Beamforming Feedback," *IEEE Networking Letters*, 2026.

Wi-Fi authenticates a station once and then trusts the session for up to an hour
or more. TrustBFI keeps checking whether the device holding an authenticated
session is still the one that authenticated, without extra frames. It reuses the
beamforming feedback information (BFI) that MIMO stations already send to the
access point for beamforming:

1. **Identity consistency.** A contrastively trained encoder embeds each window
   of `W = 10` BFI reports and scores it against a reference enrolled right after
   authentication, `s(t) = exp(-alpha * cosine_distance)`, in `[0, 1]`.
2. **Risk tracker.** A Bayesian filter fuses the score with the age of the last
   cryptographic verification into a compromise probability `q(t)`.
3. **Cost-aware trigger.** The access point runs a standard 4-way handshake only
   when `q(t) >= eta = C_v / C_c`, i.e., when waiting costs more than verifying.
   A successful handshake resets `q` and re-enrolls the reference.

The BFI is never used to grant access. It only decides *when* to reauthenticate.

## Setup

```bash
pip install -r requirements.txt
python fetch_dataset.py        # 135 files, 4.6 GB -> Dataset/vmatrices/
```

The dataset is public on Hugging Face:
[foysalhaque/BeamID](https://huggingface.co/datasets/foysalhaque/BeamID).
It holds the BFI of 15 Intel AX210 cards (802.11ax, 80 MHz) recorded at nine
locations, one `(N, 16, 250)` array per card and location
(`Dataset/vmatrices/<card>_<location>.npy`).

## Repository structure

```
TrustBFI/
├── encoder/          Step 1: BFI identity consistency
│   ├── model.py        encoder (one BFI report, 16 x 250 -> 128-d embedding)
│   ├── loss.py         supervised contrastive loss
│   ├── datagen.py      data loader and splits
│   ├── train.py        encoder training
│   └── verify.py       window embeddings, enrollment, and scores
├── algorithm/        Step 2: the TrustBFI decision system
│   ├── trustbfi.py     risk tracker, cost-aware trigger, and baselines
│   ├── calibrate.py    fits the Beta likelihoods of genuine and impostor scores
│   └── simulate.py     runs the decision system on real score traces
├── experiments/      one script per result of the paper (gen_*.py)
├── plots/            plots the figures from results/ (make_figs.sh)
├── tests/            unit tests of the decision system (no dataset needed)
├── results/          the result files used in the paper
├── figs/             the figures of the paper
├── likelihoods.json  the fitted likelihoods used in the paper
├── fetch_dataset.py  downloads the dataset
└── run_step1.sh      trains the encoder and computes the embeddings
```

All Python scripts run from the repository root as modules,
e.g., `python -m experiments.gen_fig3a`.

## Usage

```bash
./run_step1.sh                     # train the encoder, compute embeddings and scores
python -m algorithm.calibrate      # fit the likelihoods
python -m tests.test_trustbfi      # unit tests of the decision system
python -m experiments.gen_fig3a    # run an experiment (one gen_*.py per result)
cd plots && ./make_figs.sh         # plot the figures into figs/
```

The header of each `gen_*.py` names the result of the paper it produces.
`results/` and `likelihoods.json` hold the data used in the paper, so the
figures can be plotted without the dataset.

## Citation

```bibtex
@article{haque2026trustbfi,
  author  = {Haque, Khandaker Foysal and Zabin, Rifat},
  title   = {{TrustBFI}: Cost-Aware Continuous Verification for Zero-Trust {Wi-Fi} Using {MIMO} Beamforming Feedback},
  journal = {IEEE Networking Letters},
  year    = {2026}
}
```

If you use the dataset, please also cite

```bibtex
@inproceedings{haque2026beamid,
  author    = {Haque, Khandaker Foysal and Meneghello, Francesca and Restuccia, Francesco},
  title     = {{BeamID}: Domain-Adaptive Radio Fingerprinting with {MIMO} Beamforming Feedback},
  booktitle = {Proc. of IEEE NetSoft},
  year      = {2026}
}
```

## License

MIT, see [LICENSE](LICENSE). Contact: khandaker.haque@imdea.org
