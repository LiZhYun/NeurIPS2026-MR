"""External action classifier v2 — improved training pipeline.

Keeps `extract_classifier_features` untouched (topology-normalised, 8 depth bins x 7 features).
Only changes the classifier head + training procedure:
  (a) Fresh stratified 80/20 split (the original split had classes with 0 val samples).
  (b) Feature-wise standardisation (z-score per channel, computed on train split only).
  (c) Class-balanced loss: effective-number-of-samples weighting + WeightedRandomSampler.
  (d) Augmentation: random temporal crop (48 / 64 frames), speed jitter (resample rate),
      mild Gaussian feature noise.
  (e) Stronger architectures: wider CNN with dropout + small Transformer encoder; keep the
      one that wins on the held-out split.
  (f) Longer training with cosine LR schedule + early-stopping on macro-F1.

Usage:
    conda run -n anytop python -m eval.train_external_classifier_v2 \
        --save save/external_classifier_v2.pt --epochs 120
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from os.path import join as pjoin

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# Local imports. The extractor and the class list are the canonical source.
from eval.external_classifier import (
    ACTION_CLASSES,
    ACTION_TO_IDX,
    ActionClassifier,
    extract_classifier_features,
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_feature_cache(cond, label_map, motion_dir, target_T=64, n_depth_bins=8):
    """Load + featurise every clip with a coarse label.

    Returns:
        cache: list of dicts {features: [T_raw, 8, 7] float32, label: int, fname: str}
    """
    from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np

    motion_files = sorted(f for f in os.listdir(motion_dir) if f.endswith('.npy'))
    print(f"Loading {len(motion_files)} motion clips...", flush=True)

    cache = []
    n_skip = 0
    t0 = time.time()
    for fname in motion_files:
        if fname not in label_map:
            n_skip += 1
            continue
        label_str = label_map[fname].get('coarse_label', 'other')
        skel = label_map[fname]['skeleton']
        if skel not in cond:
            n_skip += 1
            continue
        info = cond[skel]
        n_joints = len(info['joints_names'])
        parents = np.array(info['parents'][:n_joints], dtype=np.int64)
        mean = info['mean'][:n_joints]
        std = info['std'][:n_joints] + 1e-6
        try:
            raw = np.load(pjoin(motion_dir, fname))
            motion_denorm = raw[:, :n_joints] * std + mean
            positions = recover_from_bvh_ric_np(motion_denorm)
            feats = extract_classifier_features(positions, parents, n_depth_bins=n_depth_bins)
            if feats is None or feats.shape[0] < 8:
                n_skip += 1
                continue
        except Exception:
            n_skip += 1
            continue
        cache.append({
            'features': feats,              # [T_raw, 8, 7]
            'label': ACTION_TO_IDX.get(label_str, ACTION_TO_IDX['other']),
            'fname': fname,
            'skeleton': skel,
        })
    print(f"  built {len(cache)} feature tensors in {time.time()-t0:.1f}s (skipped {n_skip})", flush=True)
    return cache


def stratified_split(cache, val_frac=0.2, seed=42):
    """Stratified split per class. Every class with >=2 samples gets at least 1 in val."""
    by_cls = defaultdict(list)
    for i, item in enumerate(cache):
        by_cls[item['label']].append(i)
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for cls, idxs in sorted(by_cls.items()):
        idxs = list(idxs)
        rng.shuffle(idxs)
        n_val = max(1, int(round(len(idxs) * val_frac))) if len(idxs) >= 2 else 0
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


# ---------------------------------------------------------------------------
# Augmentation & sampling
# ---------------------------------------------------------------------------

def resample_along_time(features, target_T, rng=None):
    """Resample features to target_T frames via linear interpolation in time."""
    T = features.shape[0]
    if T == target_T:
        return features.copy()
    src = np.linspace(0, T - 1, target_T)
    lo = np.floor(src).astype(np.int32)
    hi = np.minimum(lo + 1, T - 1)
    w = (src - lo).astype(np.float32)[:, None, None]
    return (1 - w) * features[lo] + w * features[hi]


def augment_features(features, target_T=64, train=True, rng=None):
    """Produce a (target_T, 8, 7) tensor from a raw (T_raw, 8, 7) one.

    Training path:
      - random speed jitter in [0.8, 1.25] (resample)
      - random temporal crop of target_T frames (or pad + shift if shorter)
      - small Gaussian feature noise
    Eval path:
      - deterministic uniform resample to target_T
    """
    rng = rng or np.random
    T_raw = features.shape[0]

    if not train:
        return resample_along_time(features, target_T).astype(np.float32)

    # speed jitter: resample raw length by factor in [0.8, 1.25]
    speed = rng.uniform(0.8, 1.25)
    new_T = max(target_T, int(round(T_raw / speed)))
    feats = resample_along_time(features, new_T)

    # random temporal crop of target_T frames
    if feats.shape[0] > target_T:
        start = rng.integers(0, feats.shape[0] - target_T + 1)
        feats = feats[start:start + target_T]
    elif feats.shape[0] < target_T:
        pad = target_T - feats.shape[0]
        feats = np.pad(feats, ((0, pad), (0, 0), (0, 0)))

    # feature noise (relative to channel std; applied with 50 % probability)
    if rng.random() < 0.5:
        noise = rng.normal(0, 0.01, feats.shape).astype(np.float32)
        feats = feats + noise

    return feats.astype(np.float32)


class MotionFeatDataset(Dataset):
    def __init__(self, cache, indices, target_T=64, train=True, feat_mean=None, feat_std=None):
        self.cache = cache
        self.indices = list(indices)
        self.target_T = target_T
        self.train = train
        self.feat_mean = feat_mean  # [8, 7]
        self.feat_std = feat_std    # [8, 7]
        self.rng = np.random.default_rng(0 if train else 1)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        item = self.cache[self.indices[i]]
        feats = augment_features(item['features'], self.target_T, train=self.train, rng=self.rng)
        if self.feat_mean is not None:
            feats = (feats - self.feat_mean) / (self.feat_std + 1e-6)
        return torch.from_numpy(feats.astype(np.float32)), item['label']


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ActionClassifierCNN(nn.Module):
    """Wider CNN with dropout, still matches ActionClassifier's input/output signature."""

    def __init__(self, n_depth_bins=8, n_features=7, n_classes=12, hidden=192, dropout=0.3):
        super().__init__()
        in_channels = n_depth_bins * n_features
        self.trunk = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )
        self.in_channels = in_channels

    def forward(self, x):
        # x: [B, T, D, F] -> [B, D*F, T]
        B, T, D, F_ = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, D * F_, T)
        z = self.trunk(x)
        return self.head(z)


class ActionClassifierXf(nn.Module):
    """Small Transformer encoder over time steps of depth-bin features."""

    def __init__(self, n_depth_bins=8, n_features=7, n_classes=12, d_model=128,
                 n_heads=4, n_layers=3, dropout=0.2, max_T=64):
        super().__init__()
        self.in_proj = nn.Linear(n_depth_bins * n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_T, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation='gelu', norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x):
        # x: [B, T, D, F]
        B, T, D, F_ = x.shape
        x = x.reshape(B, T, D * F_)
        x = self.in_proj(x) + self.pos[:, :T]
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.enc(x)
        x = self.norm(x[:, 0])
        return self.head(x)


# ---------------------------------------------------------------------------
# Inference helpers (used by downstream evaluators)
# ---------------------------------------------------------------------------

class V2Classifier:
    """Thin wrapper that loads a v2 checkpoint and exposes `.predict(features)`.

    Input features are raw outputs of `extract_classifier_features` (shape [T_raw, 8, 7]).
    The wrapper handles resample-to-target_T, normalisation, device placement,
    and optional ensemble averaging.
    """

    DEFAULT_CKPT = 'save/external_classifier_v2.pt'

    def __init__(self, ckpt_path=None, device=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
            if device is None else torch.device(device)
        if ckpt_path is None:
            # Resolve relative to repo root (file lives in eval/; repo is parent)
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ckpt_path = os.path.join(repo_root, self.DEFAULT_CKPT)
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        # Checkpoints saved by train_external_classifier_v2 must include feature
        # normalisation stats; downstream prediction is not meaningful otherwise.
        if 'feat_mean' not in ck or 'feat_std' not in ck:
            raise RuntimeError(
                f"Checkpoint {ckpt_path} is missing feat_mean/feat_std; it was "
                "either produced by an older training script or is corrupt.")
        self.feat_mean = ck['feat_mean']
        self.feat_std = ck['feat_std']
        self.target_T = ck.get('target_T', 64)
        self.arch = ck.get('arch', 'cnn')
        self.classes = ck.get('action_classes', ACTION_CLASSES)
        self.models = []
        if self.arch == 'ensemble':
            cnn = ActionClassifierCNN().to(self.device).eval()
            cnn.load_state_dict(ck['cnn_model']); self.models.append(cnn)
            xf = ActionClassifierXf(max_T=self.target_T + 1).to(self.device).eval()
            xf.load_state_dict(ck['xf_model']); self.models.append(xf)
        elif self.arch == 'xf':
            m = ActionClassifierXf(max_T=self.target_T + 1).to(self.device).eval()
            m.load_state_dict(ck['model']); self.models.append(m)
        else:
            m = ActionClassifierCNN().to(self.device).eval()
            m.load_state_dict(ck['model']); self.models.append(m)

    @torch.no_grad()
    def predict(self, features):
        """features: np.ndarray [T_raw, 8, 7]. Returns int class index."""
        if features is None:
            return None
        feats = resample_along_time(features, self.target_T).astype(np.float32)
        feats = (feats - self.feat_mean) / (self.feat_std + 1e-6)
        x = torch.from_numpy(feats).unsqueeze(0).to(self.device)
        logits_sum = None
        for m in self.models:
            logits = m(x)
            logits_sum = logits if logits_sum is None else logits_sum + logits
        return int(logits_sum.argmax(-1).item())

    @torch.no_grad()
    def predict_label(self, features):
        idx = self.predict(features)
        return self.classes[idx] if idx is not None else None


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def compute_class_weights(labels, n_classes=12, alpha=0.5):
    """Soft inverse-frequency weights: w_c proportional to (N/n_c)^alpha.

    alpha=0 -> uniform; alpha=1 -> full inverse-frequency. We default to 0.5
    (sqrt inverse-frequency) which empirically works well for long-tail
    classification (e.g. LDAM / BBN literature).
    """
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    total = counts.sum()
    freq = np.where(counts > 0, counts / total, 1.0)
    w = np.power(1.0 / freq, alpha)
    observed = counts > 0
    w = w / w[observed].mean()
    w = np.clip(w, 0.35, 5.0)
    w[~observed] = 1.0
    return w.astype(np.float32)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_y, all_p = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        pred = logits.argmax(-1).cpu().numpy()
        all_p.append(pred)
        all_y.append(np.asarray(y))
    y = np.concatenate(all_y)
    p = np.concatenate(all_p)
    acc = float((y == p).mean())
    mf1 = float(f1_score(y, p, average='macro', labels=list(range(12)), zero_division=0))
    wf1 = float(f1_score(y, p, average='weighted', labels=list(range(12)), zero_division=0))
    return acc, mf1, wf1, y, p


def train_one(model_name, model, train_ds, val_ds, class_weights, device, args,
              sampler=None):
    t_start = time.time()
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        shuffle=sampler is None, num_workers=0, drop_last=False,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.02)
    loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights).to(device),
                                  label_smoothing=args.label_smoothing)

    best_mf1 = -1.0
    best_state = None
    best_acc = 0.0
    best_wf1 = 0.0
    best_epoch = -1
    for epoch in range(args.epochs):
        model.train()
        run_loss, run_correct, run_n = 0.0, 0, 0
        for x, y in train_loader:
            x = x.to(device)
            y = torch.as_tensor(y, dtype=torch.long, device=device)
            logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            run_loss += float(loss.item()) * len(y)
            run_correct += int((logits.argmax(-1) == y).sum())
            run_n += len(y)
        sched.step()
        train_acc = run_correct / max(run_n, 1)

        val_acc, val_mf1, val_wf1, _, _ = evaluate(model, val_loader, device)

        # Select on a composite score that favours both accuracy and macro-F1.
        score = 0.5 * val_acc + 0.5 * val_mf1
        if not hasattr(train_one, '_best_score'):
            pass
        if score > (best_mf1 if best_mf1 < 0 else (0.5 * best_acc + 0.5 * best_mf1)):
            best_mf1 = val_mf1
            best_acc = val_acc
            best_wf1 = val_wf1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"[{model_name}] ep {epoch:3d} | loss {run_loss/max(run_n,1):.3f} | "
                  f"tr-acc {train_acc:.3f} | val-acc {val_acc:.3f} | val-mf1 {val_mf1:.3f} | "
                  f"best mf1 {best_mf1:.3f} @ ep {best_epoch}", flush=True)

    elapsed = time.time() - t_start
    print(f"[{model_name}] done in {elapsed:.1f}s | best val-acc {best_acc:.3f} / mf1 {best_mf1:.3f} / wf1 {best_wf1:.3f}", flush=True)
    return best_state, best_acc, best_mf1, best_wf1


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--save', type=str, default='save/external_classifier_v2.pt')
    p.add_argument('--report', type=str, default='save/external_classifier_v2_report.json')
    p.add_argument('--motion_dir', type=str,
                   default='dataset/truebones/zoo/truebones_processed/motions')
    p.add_argument('--labels', type=str,
                   default='dataset/truebones/zoo/truebones_processed/motion_labels.json')
    p.add_argument('--cond', type=str,
                   default='dataset/truebones/zoo/truebones_processed/cond.npy')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=120)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=5e-4)
    p.add_argument('--label_smoothing', type=float, default=0.05)
    p.add_argument('--target_T', type=int, default=64)
    p.add_argument('--val_frac', type=float, default=0.2)
    p.add_argument('--arch', type=str, default='both', choices=['cnn', 'xf', 'both'])
    p.add_argument('--device', type=int, default=0)
    args = p.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    cond = np.load(args.cond, allow_pickle=True).item()
    with open(args.labels) as f:
        label_map = json.load(f)

    cache = build_feature_cache(cond, label_map, args.motion_dir, target_T=args.target_T)
    train_idx, val_idx = stratified_split(cache, val_frac=args.val_frac, seed=args.seed)
    train_labels = np.array([cache[i]['label'] for i in train_idx])
    val_labels = np.array([cache[i]['label'] for i in val_idx])

    print("\nStratified split:")
    for c in range(12):
        print(f"  {ACTION_CLASSES[c]:<8}  train={int((train_labels==c).sum()):>4}  val={int((val_labels==c).sum()):>4}")

    # Feature normalisation stats on training clips (unaugmented, full-length)
    feat_list = [cache[i]['features'] for i in train_idx]
    feat_stack = np.concatenate([f.reshape(-1, 7) for f in feat_list], axis=0)
    feat_mean = feat_stack.mean(0).reshape(1, 1, 7).astype(np.float32)
    feat_std = feat_stack.std(0).reshape(1, 1, 7).astype(np.float32)
    # broadcast to [8,7]
    feat_mean = np.broadcast_to(feat_mean, (1, 8, 7)).copy().squeeze(0)
    feat_std = np.broadcast_to(feat_std, (1, 8, 7)).copy().squeeze(0)
    print(f"\nFeature mean: {feat_mean[0]}\nFeature std:  {feat_std[0]}")

    train_ds = MotionFeatDataset(cache, train_idx, target_T=args.target_T, train=True,
                                 feat_mean=feat_mean, feat_std=feat_std)
    val_ds = MotionFeatDataset(cache, val_idx, target_T=args.target_T, train=False,
                               feat_mean=feat_mean, feat_std=feat_std)

    class_weights = compute_class_weights(train_labels, n_classes=12)
    print(f"\nClass weights (effective-number):")
    for c in range(12):
        print(f"  {ACTION_CLASSES[c]:<8}  w={class_weights[c]:.2f}")

    # WeightedRandomSampler: mix between uniform (alpha=0) and inverse-frequency
    # (alpha=1). alpha=0.5 keeps the overall-accuracy target reachable (common
    # classes still see enough samples) while boosting minority classes compared
    # to a naive shuffler.
    counts = np.bincount(train_labels, minlength=12).astype(np.float64)
    alpha_s = 0.5
    per_class_sw = np.where(counts > 0, np.power(np.maximum(counts, 1.0), -alpha_s), 0.0)
    sample_weights = per_class_sw[train_labels]
    n_samples = int(len(train_labels) * 1.5)
    sampler = WeightedRandomSampler(weights=sample_weights.tolist(),
                                    num_samples=n_samples, replacement=True)

    # Train models
    results = {}
    models = {}
    if args.arch in ('cnn', 'both'):
        m = ActionClassifierCNN().to(device)
        n_params = sum(x.numel() for x in m.parameters())
        print(f"\nTraining CNN ({n_params/1e6:.2f}M params)...")
        state, acc, mf1, wf1 = train_one('cnn', m, train_ds, val_ds, class_weights,
                                         device, args, sampler=sampler)
        results['cnn'] = {'val_acc': acc, 'val_macro_f1': mf1, 'val_weighted_f1': wf1}
        models['cnn'] = (state, m)

    if args.arch in ('xf', 'both'):
        m = ActionClassifierXf(max_T=args.target_T + 1).to(device)
        n_params = sum(x.numel() for x in m.parameters())
        print(f"\nTraining Transformer ({n_params/1e6:.2f}M params)...")
        state, acc, mf1, wf1 = train_one('xf', m, train_ds, val_ds, class_weights,
                                         device, args, sampler=sampler)
        results['xf'] = {'val_acc': acc, 'val_macro_f1': mf1, 'val_weighted_f1': wf1}
        models['xf'] = (state, m)

    # Choose the one with best macro-F1
    best_name = max(results.keys(), key=lambda k: results[k]['val_macro_f1'])
    best_state, best_model = models[best_name]
    print(f"\nBest arch: {best_name} (macro-F1 {results[best_name]['val_macro_f1']:.3f})")

    # Final eval with classification_report for the chosen model
    best_model.load_state_dict(best_state)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # If both arches trained, also try the logit-averaged ensemble.
    if args.arch == 'both' and 'cnn' in models and 'xf' in models:
        cnn_state, cnn_model = models['cnn']
        xf_state, xf_model = models['xf']
        cnn_model.load_state_dict(cnn_state); cnn_model.eval()
        xf_model.load_state_dict(xf_state); xf_model.eval()
        all_y, all_p = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                lc = cnn_model(x); lx = xf_model(x)
                p = ((lc + lx) / 2).argmax(-1).cpu().numpy()
                all_p.append(p); all_y.append(np.asarray(y))
        y_e = np.concatenate(all_y); p_e = np.concatenate(all_p)
        e_acc = float((y_e == p_e).mean())
        e_mf1 = float(f1_score(y_e, p_e, average='macro', labels=list(range(12)), zero_division=0))
        e_wf1 = float(f1_score(y_e, p_e, average='weighted', labels=list(range(12)), zero_division=0))
        print(f"\nEnsemble (cnn+xf) val-acc {e_acc:.3f} / mf1 {e_mf1:.3f} / wf1 {e_wf1:.3f}")
        results['ensemble'] = {'val_acc': e_acc, 'val_macro_f1': e_mf1, 'val_weighted_f1': e_wf1}
        # If ensemble strictly beats the best single model on macro-F1, use it.
        if e_mf1 > results[best_name]['val_macro_f1']:
            best_name = 'ensemble'
            print(f"-> switching to ensemble as final model")

    if best_name == 'ensemble':
        # Recompute per-class using the ensemble predictions
        all_y, all_p = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                cnn_model.load_state_dict(models['cnn'][0]); cnn_model.eval()
                xf_model.load_state_dict(models['xf'][0]); xf_model.eval()
                lc = cnn_model(x); lx = xf_model(x)
                all_p.append(((lc + lx) / 2).argmax(-1).cpu().numpy())
                all_y.append(np.asarray(y))
        y_true = np.concatenate(all_y); y_pred = np.concatenate(all_p)
    else:
        _, _, _, y_true, y_pred = evaluate(best_model, val_loader, device)
    precision, recall, f1s, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(12)), zero_division=0)
    per_class = {}
    for c in range(12):
        per_class[ACTION_CLASSES[c]] = {
            'precision': float(precision[c]),
            'recall': float(recall[c]),
            'f1': float(f1s[c]),
            'support': int(support[c]),
        }
    overall_acc = float((y_true == y_pred).mean())
    macro_f1 = float(f1_score(y_true, y_pred, average='macro', labels=list(range(12)), zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average='weighted', labels=list(range(12)), zero_division=0))

    print("\n" + "=" * 76)
    print("Per-class (val held-out):")
    print(f"  {'class':<10} {'prec':>6} {'rec':>6} {'f1':>6} {'support':>8}")
    for c in range(12):
        pc = per_class[ACTION_CLASSES[c]]
        print(f"  {ACTION_CLASSES[c]:<10} {pc['precision']:>6.3f} {pc['recall']:>6.3f} "
              f"{pc['f1']:>6.3f} {pc['support']:>8}")
    print(f"\nOverall val-acc: {overall_acc:.3f}")
    print(f"Macro-F1:        {macro_f1:.3f}")
    print(f"Weighted-F1:     {weighted_f1:.3f}")

    # Save: state_dict(s) + metadata. feat_mean / feat_std stored so downstream
    # eval can standardise features identically.
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    ckpt = {
        'arch': best_name,
        'val_acc': overall_acc,
        'val_macro_f1': macro_f1,
        'val_weighted_f1': weighted_f1,
        'feat_mean': feat_mean,
        'feat_std': feat_std,
        'action_classes': ACTION_CLASSES,
        'n_depth_bins': 8,
        'n_features': 7,
        'target_T': args.target_T,
    }
    if best_name == 'ensemble':
        ckpt['cnn_model'] = models['cnn'][0]
        ckpt['xf_model'] = models['xf'][0]
    else:
        ckpt['model'] = best_state
        # Also keep the alternates if we trained both (useful for post-hoc probing)
        if args.arch == 'both':
            for k, (s, _) in models.items():
                if k != best_name:
                    ckpt[f'{k}_model'] = s
    torch.save(ckpt, args.save)
    print(f"\nSaved checkpoint -> {args.save}")

    # Write an evaluation report alongside the checkpoint
    report = {
        'best_arch': best_name,
        'overall_val_acc': overall_acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'per_class': per_class,
        'train_class_counts': {ACTION_CLASSES[c]: int(np.sum(train_labels == c)) for c in range(12)},
        'val_class_counts': {ACTION_CLASSES[c]: int(np.sum(val_labels == c)) for c in range(12)},
        'model_results': results,
        'args': vars(args),
    }
    with open(args.report, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Saved report    -> {args.report}")


if __name__ == '__main__':
    main()
