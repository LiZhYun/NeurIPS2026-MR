"""PyTorch dataset for pre-encoded invariant motion representations."""
from __future__ import annotations

import json
import os
from typing import Optional, Set

import numpy as np
import torch
from torch.utils.data import Dataset

INVARIANT_DIR = "dataset/truebones/zoo/invariant_reps"

# Hold-out skeletons for evaluation — chosen for morphological diversity:
# 2 quadrupeds (Cat, Elephant), 2 arthropods (Spider, Crab), 2 bipeds (Trex, Raptor),
# 1 flying (Buzzard), 1 reptile (Alligator), 1 snake (Anaconda), 1 small (Rat)
TEST_SKELETONS = frozenset({
    "Cat", "Elephant", "Spider", "Crab", "Trex",
    "Raptor", "Buzzard", "Alligator", "Anaconda", "Rat",
})


class InvariantMotionDataset(Dataset):
    """Loads pre-encoded [T, 32, 8] invariant reps, crops to fixed window."""

    def __init__(self, window: int = 40, data_dir: str = INVARIANT_DIR,
                 split: str = "train", test_skeletons: Set[str] = TEST_SKELETONS):
        self.window = window
        manifest_path = os.path.join(data_dir, "manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)

        self.clips = []
        self.clip_names = []
        for skel_name, info in manifest["skeletons"].items():
            is_test = skel_name in test_skeletons
            if split == "train" and is_test:
                continue
            if split == "test" and not is_test:
                continue
            npz_path = os.path.join(data_dir, f"{skel_name}.npz")
            data = np.load(npz_path, allow_pickle=True)
            for clip_name in info["clips"]:
                arr = data[clip_name]
                if arr.shape[0] >= window:
                    self.clips.append(arr)
                    self.clip_names.append(clip_name)

        print(f"InvariantMotionDataset({split}): {len(self.clips)} clips (≥{window} frames)")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = self.clips[idx]
        T = clip.shape[0]
        start = np.random.randint(0, T - self.window + 1)
        window = clip[start:start + self.window]
        return torch.from_numpy(window.copy()).float()
