# datagen.py
# Each file: (N, 250, 4, 2) complex -> returns x: (16,250), y: station_id, loc: location_id
# 16 = 4(Tx) * 2(Rx) * 2(Re/Im)
# x is shaped for Conv1D: (C, L) = (16, 250)

import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset, Subset


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_filename(path):
    name = os.path.basename(path).replace(".npy", "")
    mac, loc = name.split("_")
    return mac.lower(), int(loc)


class VMatDataset(Dataset):
    def __init__(self, root_dir, locations=None, devices=None, normalize=True, mmap=True):
        self.normalize = normalize
        self.mmap = mmap
        self.locations = set(locations) if locations is not None else None
        # device holdout: keep only these mac prefixes (open-set verification needs unseen devices)
        self.devices = {d.lower() for d in devices} if devices is not None else None

        self.files = sorted(glob.glob(os.path.join(root_dir, "*.npy")))

        # station label map from selected locations
        macs = []
        for f in self.files:
            mac, loc = parse_filename(f)
            if self._keep(mac, loc):
                macs.append(mac)
        self.mac_to_y = {m: i for i, m in enumerate(sorted(set(macs)))}

        # flat index: (file, sample_idx, y, loc)
        self.index = []
        self._cache = {}

        for f in self.files:
            mac, loc = parse_filename(f)
            if not self._keep(mac, loc):
                continue

            arr = np.load(f, allow_pickle=True)
            n = arr.shape[0]
            y = self.mac_to_y[mac]

            for i in range(n):
                self.index.append((f, i, y, loc))

    def _keep(self, mac, loc):
        if self.locations is not None and loc not in self.locations:
            return False
        if self.devices is not None and mac not in self.devices:
            return False
        return True

    def __len__(self):
        return len(self.index)

    def _get_arr(self, f):
        if f not in self._cache:
            self._cache[f] = np.load(f, mmap_mode="r" if self.mmap else None)
        return self._cache[f]

    def __getitem__(self, idx):
        f, i, y, loc = self.index[idx]
        arr = self._get_arr(f)

        # packet: (250,4,2) complex
        x = arr[i]

        # split complex -> real/imag
        xr = np.real(x).astype(np.float32, copy=False)   # (250,4,2)
        xi = np.imag(x).astype(np.float32, copy=False)   # (250,4,2)

        # stack real/imag as a new axis: (2,250,4,2)
        x = np.stack([xr, xi], axis=0)

        # reorder to (Re/Im, Rx, Tx, Subc) => (2,2,4,250)
        x = np.transpose(x, (0, 3, 2, 1))

        # flatten channels: 2*2*4=16, keep subchannels=250 => (16,250)
        x = x.reshape(16, 250)

        if self.normalize:
            x = x / (np.linalg.norm(x.reshape(-1)) + 1e-8)

        return torch.from_numpy(x), torch.tensor(y), torch.tensor(loc)


def stratified_split(ds, val_ratio=0.2, seed=0):
    set_seed(seed)
    by_y = {}
    for idx, (_, _, y, _) in enumerate(ds.index):
        by_y.setdefault(y, []).append(idx)

    train_idx, val_idx = [], []
    for y, idxs in by_y.items():
        random.shuffle(idxs)
        n_val = int(round(len(idxs) * val_ratio))
        val_idx += idxs[:n_val]
        train_idx += idxs[n_val:]

    return Subset(ds, train_idx), Subset(ds, val_idx)


def fewshot_split(ds, k_per_class=5, seed=0):
    set_seed(seed)
    by_y = {}
    for idx, (_, _, y, _) in enumerate(ds.index):
        by_y.setdefault(y, []).append(idx)

    adapt_idx, test_idx = [], []
    for y, idxs in by_y.items():
        random.shuffle(idxs)
        adapt_idx += idxs[:k_per_class]
        test_idx += idxs[k_per_class:]

    return Subset(ds, adapt_idx), Subset(ds, test_idx)


def make_splits(root_dir, source_locs, target_locs, k_per_class=5, val_ratio=0.2, seed=0, devices=None):
    src = VMatDataset(root_dir, locations=source_locs, devices=devices, normalize=True, mmap=True)
    tgt = VMatDataset(root_dir, locations=target_locs, devices=devices, normalize=True, mmap=True)

    src_train, src_val = stratified_split(src, val_ratio=val_ratio, seed=seed)
    tgt_adapt, tgt_test = fewshot_split(tgt, k_per_class=k_per_class, seed=seed)

    return src_train, src_val, tgt_adapt, tgt_test, src.mac_to_y
