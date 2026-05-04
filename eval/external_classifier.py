"""External action classifier — independent evaluator for Track B.

Train a simple classifier on REAL Truebones clips with motion-class labels.
Independence: trained only on real data, never sees our synthetic edits or our analytic ψ.
Features: simple per-frame joint statistics (different from our ψ to avoid circularity):
  - Per-joint mean velocity (independent of joint count via depth-binned aggregation)
  - Per-joint mean position vs centroid
  - Aggregated using a small 1D CNN over time

Used for action transfer accuracy in Track B.

Usage:
    # Train
    conda run -n anytop python -m eval.external_classifier --train --save save/external_classifier.pt
    # Eval generated motion
    conda run -n anytop python -m eval.external_classifier --eval generated.npz --ckpt save/external_classifier.pt
"""
import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from os.path import join as pjoin


ACTION_CLASSES = ['walk', 'run', 'idle', 'attack', 'fly', 'swim', 'jump',
                  'turn', 'die', 'eat', 'getup', 'other']
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_CLASSES)}


def extract_classifier_features(positions, parents, n_depth_bins=8):
    """Extract topology-normalized features for action classification.

    positions: [T, J, 3] — global joint positions (denormalized)
    parents: [J] parents

    Returns: [T-1, n_depth_bins, 6] feature tensor (independent of J)
    """
    T, J, _ = positions.shape
    if T < 2:
        return None

    # Compute geodesic depth per joint
    depth = np.zeros(J)
    for j in range(J):
        d = 0
        cur = j
        while parents[cur] != cur and parents[cur] >= 0 and d < J:
            d += 1
            cur = parents[cur]
        depth[j] = d
    if depth.max() < 1:
        return None
    depth_norm = depth / depth.max()
    depth_bin = np.clip((depth_norm * n_depth_bins).astype(int), 0, n_depth_bins - 1)

    # Per-frame joint velocity (relative to centroid)
    centroid = positions.mean(axis=1, keepdims=True)  # [T, 1, 3]
    pos_rel = positions - centroid  # [T, J, 3]
    vel = np.diff(pos_rel, axis=0)  # [T-1, J, 3]
    vel_mag = np.linalg.norm(vel, axis=-1)  # [T-1, J]

    # Per-frame joint height relative to lowest joint
    height = positions[:, :, 1]  # [T, J]
    height_rel = height - height.min(axis=1, keepdims=True)  # [T, J]

    # Per-frame absolute joint speed (centroid speed)
    centroid_speed = np.linalg.norm(np.diff(centroid.squeeze(1), axis=0), axis=-1)  # [T-1]

    # Aggregate by depth bin: per (frame, depth_bin) → 6 features
    # (mean_vel, max_vel, mean_height, mean_centroid_dist, max_centroid_dist, count_norm)
    features = np.zeros((T-1, n_depth_bins, 6))
    centroid_dist = np.linalg.norm(pos_rel, axis=-1)  # [T, J]

    for b in range(n_depth_bins):
        mask = depth_bin == b
        if not mask.any():
            continue
        features[:, b, 0] = vel_mag[:, mask].mean(axis=1)
        features[:, b, 1] = vel_mag[:, mask].max(axis=1)
        features[:, b, 2] = height_rel[1:, mask].mean(axis=1)
        features[:, b, 3] = centroid_dist[1:, mask].mean(axis=1)
        features[:, b, 4] = centroid_dist[1:, mask].max(axis=1)
        features[:, b, 5] = mask.sum() / J  # fraction of joints in this bin

    # Append global centroid speed channel (broadcast across bins)
    centroid_chan = np.repeat(centroid_speed[:, None, None], n_depth_bins, axis=1)
    features = np.concatenate([features, centroid_chan], axis=-1)  # [T-1, 8 bins, 7]

    return features.astype(np.float32)


class ActionClassifier(nn.Module):
    """1D CNN over time × small MLP over depth bins, then linear head."""
    def __init__(self, n_depth_bins=8, n_features=7, n_classes=12, hidden=128):
        super().__init__()
        # Input: [B, T, n_depth_bins, n_features] → reshape to [B, n_depth_bins*n_features, T]
        in_channels = n_depth_bins * n_features
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2, stride=2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2, stride=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_classes),
        )
        self.in_channels = in_channels

    def forward(self, x):
        # x: [B, T, n_depth_bins, n_features] → [B, n_depth_bins*n_features, T]
        B, T, D, F_ = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, D * F_, T)
        return self.conv(x)


def build_dataset():
    """Load all REAL Truebones clips with action labels, extract features."""
    from data_loaders.truebones.truebones_utils.get_opt import get_opt
    from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np

    opt = get_opt(0)
    cond_dict = np.load(opt.cond_file, allow_pickle=True).item()
    with open('dataset/truebones/zoo/truebones_processed/motion_labels.json') as f:
        label_map = json.load(f)
    with open('dataset/truebones/zoo/truebones_processed/train_val_split.json') as f:
        split = json.load(f)

    motion_files = sorted(f for f in os.listdir(opt.motion_dir) if f.endswith('.npy'))
    print(f"Building dataset from {len(motion_files)} clips")

    train_X, train_y = [], []
    val_X, val_y = [], []
    n_skip = 0
    for fname in motion_files:
        if fname not in label_map:
            n_skip += 1
            continue
        label_str = label_map[fname].get('coarse_label', 'other')
        skel = label_map[fname]['skeleton']
        if skel not in cond_dict:
            n_skip += 1
            continue

        info = cond_dict[skel]
        n_joints = len(info['joints_names'])
        parents = np.array(info['parents'][:n_joints], dtype=np.int64)
        mean = info['mean'][:n_joints]
        std = info['std'][:n_joints] + 1e-6

        try:
            raw = np.load(pjoin(opt.motion_dir, fname))
            motion_denorm = raw[:, :n_joints] * std + mean
            positions = recover_from_bvh_ric_np(motion_denorm)
            features = extract_classifier_features(positions, parents)
            if features is None:
                n_skip += 1
                continue
            # Resample to fixed length 64 frames (TARGET_FRAMES)
            T_feat = features.shape[0]
            if T_feat < 64:
                features = np.pad(features, ((0, 64 - T_feat), (0, 0), (0, 0)))
            else:
                indices = np.linspace(0, T_feat - 1, 64).astype(int)
                features = features[indices]
        except Exception as e:
            n_skip += 1
            continue

        label = ACTION_TO_IDX.get(label_str, ACTION_TO_IDX['other'])
        if fname in set(split['train']):
            train_X.append(features)
            train_y.append(label)
        else:
            val_X.append(features)
            val_y.append(label)

    print(f"  Train: {len(train_X)}, Val: {len(val_X)}, Skipped: {n_skip}")
    return (np.stack(train_X), np.array(train_y),
            np.stack(val_X), np.array(val_y))


def train_classifier(args):
    from utils.fixseed import fixseed
    from utils import dist_util
    fixseed(args.seed)
    dist_util.setup_dist(args.device)
    device = dist_util.dev()

    train_X, train_y, val_X, val_y = build_dataset()
    print(f"Class distribution (train): {np.bincount(train_y, minlength=12)}")

    train_X_t = torch.tensor(train_X, dtype=torch.float32)
    train_y_t = torch.tensor(train_y, dtype=torch.long)
    val_X_t = torch.tensor(val_X, dtype=torch.float32)
    val_y_t = torch.tensor(val_y, dtype=torch.long)

    model = ActionClassifier()
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    n_epochs = 50
    bs = 32
    best_val_acc = 0
    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(len(train_X_t))
        ep_loss, ep_correct = 0, 0
        for i in range(0, len(perm), bs):
            idx = perm[i:i+bs]
            x = train_X_t[idx].to(device)
            y = train_y_t[idx].to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item() * len(idx)
            ep_correct += (logits.argmax(-1) == y).sum().item()
        train_acc = ep_correct / len(train_X_t)

        model.eval()
        with torch.no_grad():
            v_correct = 0
            for i in range(0, len(val_X_t), bs):
                x = val_X_t[i:i+bs].to(device)
                y = val_y_t[i:i+bs].to(device)
                v_correct += (model(x).argmax(-1) == y).sum().item()
            val_acc = v_correct / len(val_X_t)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({'model': model.state_dict(), 'val_acc': val_acc, 'epoch': epoch},
                       args.save)
        if epoch % 5 == 0 or epoch == n_epochs - 1:
            print(f"  ep {epoch}: train_acc={train_acc:.3f} val_acc={val_acc:.3f} best={best_val_acc:.3f}")

    print(f"\nBest val acc: {best_val_acc:.3f} → saved to {args.save}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train', action='store_true')
    p.add_argument('--save', default='save/external_classifier.pt')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=int, default=0)
    args = p.parse_args()

    if args.train:
        train_classifier(args)
    else:
        print("Use --train to train the external classifier")


if __name__ == '__main__':
    main()
