"""Paired invariant rep dataset for supervised cross-skeleton CFM training.

For each action with clips on ≥2 skeletons, creates (source, target) pairs
where source and target are from different skeletons performing the same action.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Set

import numpy as np
import torch
from torch.utils.data import Dataset

from model.skel_blind.invariant_dataset import INVARIANT_DIR, TEST_SKELETONS


class PairedInvariantDataset(Dataset):

    def __init__(self, window: int = 40, data_dir: str = INVARIANT_DIR,
                 split: str = "train", test_skeletons: Set[str] = TEST_SKELETONS):
        self.window = window

        manifest_path = os.path.join(data_dir, "manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)

        clips_by_action = defaultdict(list)
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
                if arr.shape[0] < window:
                    continue
                parts = clip_name.split("___")
                if len(parts) >= 2:
                    action = parts[1].rsplit("_", 1)[0].lower()
                else:
                    action = clip_name.lower()
                clips_by_action[action].append({
                    "skel": skel_name,
                    "clip": arr,
                })

        self.pairs = []
        for action, clips in clips_by_action.items():
            skels = list(set(c["skel"] for c in clips))
            if len(skels) < 2:
                continue
            for i, src in enumerate(clips):
                for j, tgt in enumerate(clips):
                    if src["skel"] != tgt["skel"]:
                        self.pairs.append((src["clip"], tgt["clip"]))

        print(f"PairedInvariantDataset({split}): {len(self.pairs)} cross-skeleton pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_clip, tgt_clip = self.pairs[idx]

        src_T = src_clip.shape[0]
        tgt_T = tgt_clip.shape[0]
        s0 = np.random.randint(0, src_T - self.window + 1)
        frac = s0 / max(src_T - self.window, 1)
        t0 = min(int(frac * max(tgt_T - self.window, 1)),
                 tgt_T - self.window)

        src_w = src_clip[s0:s0 + self.window]
        tgt_w = tgt_clip[t0:t0 + self.window]

        return (torch.from_numpy(src_w.copy()).float(),
                torch.from_numpy(tgt_w.copy()).float())
