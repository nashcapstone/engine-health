"""PyTorch dataloaders over the saved splits.

    from data.loaders import get_dataloaders
    loaders = get_dataloaders(batch_size=256)
    for batch in loaders["train"]:      # dict of tensors in the contract 1 format
        ...
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import config
from contracts import BATCH_KEYS
from data.preprocess import OUT_DIR, SPLITS, load_split


class EngineWindows(Dataset):
    def __init__(self, batch):
        self.tensors = {k: torch.from_numpy(np.asarray(batch[k])) for k in BATCH_KEYS}

    def __len__(self):
        return len(self.tensors["X"])

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.tensors.items()}


def get_dataloaders(batch_size=256, out_dir=OUT_DIR, seed=config.SEED):
    """Shuffles train only. Default collate stacks the dict, so each batch passes check_batch."""
    generator = torch.Generator().manual_seed(seed)
    return {
        name: DataLoader(
            EngineWindows(load_split(name, out_dir)),
            batch_size=batch_size,
            shuffle=name == "train",
            generator=generator if name == "train" else None,
        )
        for name in SPLITS
    }
