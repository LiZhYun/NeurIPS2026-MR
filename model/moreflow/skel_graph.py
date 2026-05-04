"""Skeleton-morphology encoder for MoReFlow Stage B conditioning.

Per-skel features (cached once at startup from cond.npy):
  - Per-joint: [offset_xyz(3), depth_normalized(1), is_leaf(1), is_root(1)] ∈ R^6
  - DeepSets MLP → mean-pool → R^64
  - Concat aggregates [J, max_depth, limb_len_mean, limb_len_std] ∈ R^4 → MLP → R^128

This is needed mainly for the inductive ablation (10 test_test skels held out from Stage B
training; the model sees them only via these graph features). For the transductive primary
run, it's an auxiliary signal.

Caches the per-skel raw feature tensor; the encoder MLP is part of the flow transformer.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def compute_joint_depths(parents):
    """BFS distance from root (joint 0) for each joint. parents: [J] (root has parent -1)."""
    J = len(parents)
    depths = -np.ones(J, dtype=np.int64)
    depths[0] = 0
    # Iterate until stable
    changed = True
    while changed:
        changed = False
        for j in range(1, J):
            p = int(parents[j])
            if depths[j] < 0 and depths[p] >= 0:
                depths[j] = depths[p] + 1
                changed = True
    # Any remaining -1 means disconnected; set to max depth + 1
    if (depths < 0).any():
        depths[depths < 0] = depths.max() + 1
    return depths


def compute_is_leaf(parents):
    """Joint j is a leaf iff no other joint has parent == j."""
    J = len(parents)
    has_child = np.zeros(J, dtype=bool)
    for j in range(J):
        p = int(parents[j])
        if p >= 0 and p < J:
            has_child[p] = True
    return (~has_child).astype(np.float32)


def per_joint_features(offsets, parents, max_depth_global=20):
    """Build [J, 6] per-joint feature.

    Features: [offset_x, offset_y, offset_z, depth_normalized, is_leaf, is_root]
    """
    J = len(parents)
    depths = compute_joint_depths(parents).astype(np.float32)
    is_leaf = compute_is_leaf(parents)
    is_root = np.zeros(J, dtype=np.float32)
    is_root[0] = 1.0
    depth_normalized = depths / max(max_depth_global, depths.max() + 1e-6)
    feats = np.concatenate([
        offsets.astype(np.float32),                            # [J, 3]
        depth_normalized[:, None],                              # [J, 1]
        is_leaf[:, None],                                       # [J, 1]
        is_root[:, None],                                       # [J, 1]
    ], axis=1)                                                  # [J, 6]
    return feats


def aggregate_features(offsets, parents):
    """Build R^4 aggregate features for the skel.

    [n_joints, max_depth, limb_len_mean, limb_len_std]
    """
    J = len(parents)
    depths = compute_joint_depths(parents)
    limb_lens = np.linalg.norm(offsets[1:], axis=-1)            # [J-1] (skip root which has 0 offset)
    return np.array([
        float(J),
        float(depths.max()),
        float(limb_lens.mean()) if len(limb_lens) > 0 else 0.0,
        float(limb_lens.std()) if len(limb_lens) > 0 else 0.0,
    ], dtype=np.float32)


def build_skel_features(cond_dict):
    """Pre-compute per-skel raw features. Returns dict: skel → {'per_joint': Tensor[J, 6], 'agg': Tensor[4]}.

    Per-joint features have variable J. Caller pads + masks per batch.
    """
    out = {}
    for skel, c in cond_dict.items():
        offsets = np.asarray(c['offsets'], dtype=np.float32)
        parents = np.asarray(c['parents'], dtype=np.int64)
        out[skel] = {
            'per_joint': torch.from_numpy(per_joint_features(offsets, parents)),
            'agg': torch.from_numpy(aggregate_features(offsets, parents)),
            'n_joints': len(parents),
        }
    return out


def pad_to_max_joints(per_joint_feats, max_joints, pad_value=0.0):
    """per_joint_feats: [J, 6]; pad to [max_joints, 6] and return mask [max_joints]."""
    J, F = per_joint_feats.shape
    if J >= max_joints:
        return per_joint_feats[:max_joints], torch.ones(max_joints)
    out = torch.full((max_joints, F), pad_value, dtype=per_joint_feats.dtype)
    out[:J] = per_joint_feats
    mask = torch.zeros(max_joints)
    mask[:J] = 1.0
    return out, mask


class SkelGraphEncoder(nn.Module):
    """DeepSets encoder over per-joint features + concat with aggregate features.

    Input:
      per_joint: [B, J_max, 6]
      mask:      [B, J_max]  (1 = valid, 0 = padding)
      agg:       [B, 4]
    Output:
      skel_emb:  [B, d_out=128]
    """
    def __init__(self, d_in=6, d_hidden=64, d_agg=4, d_out=128):
        super().__init__()
        self.joint_mlp = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden),
        )
        self.agg_mlp = nn.Sequential(
            nn.Linear(d_agg, d_hidden),
            nn.SiLU(),
        )
        self.out_mlp = nn.Sequential(
            nn.Linear(d_hidden + d_hidden, d_out),
            nn.SiLU(),
            nn.Linear(d_out, d_out),
        )

    def forward(self, per_joint, mask, agg):
        h = self.joint_mlp(per_joint)                                       # [B, J_max, d_hidden]
        h_masked = h * mask.unsqueeze(-1)                                   # zero-out padding
        # Mean over valid joints (avoid divide-by-zero)
        n_valid = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        h_pool = h_masked.sum(dim=1) / n_valid                              # [B, d_hidden]
        agg_h = self.agg_mlp(agg)                                            # [B, d_hidden]
        skel_emb = self.out_mlp(torch.cat([h_pool, agg_h], dim=-1))          # [B, d_out]
        return skel_emb


if __name__ == '__main__':
    cond = np.load(PROJECT_ROOT / 'dataset/truebones/zoo/truebones_processed/cond.npy',
                   allow_pickle=True).item()
    feats = build_skel_features(cond)
    print(f"{len(feats)} skels processed")
    for skel in ['Horse', 'Bat', 'Anaconda', 'Cat']:
        if skel in feats:
            f = feats[skel]
            print(f"  {skel}: J={f['n_joints']}, per_joint shape={f['per_joint'].shape}, "
                  f"agg={f['agg'].numpy()}")

    # Smoke test the encoder
    enc = SkelGraphEncoder()
    max_J = max(f['n_joints'] for f in feats.values())
    print(f"\nMax J across all skels: {max_J}")
    pj_padded, mask = pad_to_max_joints(feats['Horse']['per_joint'], max_J)
    pj_padded2, mask2 = pad_to_max_joints(feats['Bat']['per_joint'], max_J)
    batch_pj = torch.stack([pj_padded, pj_padded2])
    batch_mask = torch.stack([mask, mask2])
    batch_agg = torch.stack([feats['Horse']['agg'], feats['Bat']['agg']])
    out = enc(batch_pj, batch_mask, batch_agg)
    print(f"  encoder output: {out.shape} (expected [2, 128])")
    print(f"  Horse emb mean/std: {out[0].mean():.4f}/{out[0].std():.4f}")
    print(f"  Bat   emb mean/std: {out[1].mean():.4f}/{out[1].std():.4f}")
