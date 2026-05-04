"""Extract Truebones motions to numpy format for Set Transformer evaluator training.

Produces a .npy dict with:
  motions: [N, J_max, 13, T_max]  float32 — normalized motion tensors
  masks:   [N, J_max]             bool    — True = real joint
  names:   [N]                    str     — "ObjectType_motionname"
  object_types: [N]               str     — skeleton type label

Usage:
  python eval/extract_eval_data.py --split train --out eval/data/truebones_train.npy
  python eval/extract_eval_data.py --split val   --out eval/data/truebones_val.npy
"""

import argparse
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader


def collate_identity(batch):
    return batch


def extract(split, out_path, num_frames=40, temporal_window=31, objects_subset='all'):
    # Import here so PYTHONPATH is already set when this runs as a module
    from data_loaders.truebones.data.dataset_conditioned import TruebonesConditioned
    from data_loaders.tensors_conditioned import truebones_batch_collate_conditioned

    print(f'Loading {split} split (subset={objects_subset})...')
    dataset_wrapper = TruebonesConditioned(
        split=split,
        num_frames=num_frames,
        temporal_window=temporal_window,
        t5_name='t5-base',
        balanced=False,
        objects_subset=objects_subset,
    )
    motion_ds = dataset_wrapper.motion_dataset

    # Access the underlying dataset directly (no augmentation, deterministic)
    N = len(motion_ds)
    print(f'  {N} samples found')

    max_joints  = motion_ds.opt.max_joints   # 143
    max_frames  = motion_ds.opt.max_motion_length

    motions      = np.zeros((N, max_joints, 13, max_frames), dtype=np.float32)
    masks        = np.zeros((N, max_joints),                  dtype=bool)
    names        = []
    object_types = []

    for i in range(N):
        if i % 100 == 0:
            print(f'  {i}/{N}', flush=True)

        item = motion_ds[i]
        # item[0] = motion [T, J, 13]  (normalized, padded to max_frames)
        # item[8] = object_type
        # item[13] = max_joints (scalar)
        mot        = item[0]        # [T, J, 13]
        n_joints   = mot.shape[1]
        object_type = item[8]
        name_idx   = motion_ds.pointer + i if not motion_ds.balanced else i
        # name_list may wrap; use modulo
        name = motion_ds.name_list[name_idx % len(motion_ds.name_list)]

        # Permute to [J, 13, T] then pad
        mot_t = np.transpose(mot, (1, 2, 0))  # [J, 13, T]
        motions[i, :n_joints, :, :mot_t.shape[2]] = mot_t
        masks[i, :n_joints] = True
        names.append(name)
        object_types.append(object_type)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    data = {
        'motions':      motions,        # [N, J_max, 13, T_max]
        'masks':        masks,          # [N, J_max]
        'names':        np.array(names),
        'object_types': np.array(object_types),
    }
    np.save(out_path, data)
    print(f'Saved {N} samples → {out_path}')
    print(f'  motions: {motions.shape}  masks: {masks.shape}')
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split',          default='train',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--out',            default='eval/data/truebones_train.npy')
    parser.add_argument('--num_frames',     type=int, default=40)
    parser.add_argument('--temporal_window', type=int, default=31)
    parser.add_argument('--objects_subset', default='all')
    args = parser.parse_args()

    extract(args.split, args.out, args.num_frames, args.temporal_window,
            args.objects_subset)
