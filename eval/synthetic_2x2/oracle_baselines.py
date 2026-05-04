"""Synthetic 2x2 oracle baselines.

Computes the recovery error of 4 baselines that the trained SmallFlowGen should be
compared against, per Codex R2 weakness #3:
  1. zero — predict zero
  2. random-noise — predict standard Gaussian
  3. random-same-action — randomly sample a clip with the same (skel_b, action) cell
  4. source-blind cell-mean — predict E[T*(x_a) | skel_b, action] computed from training pairs
  5. oracle-with-x_a — predict T*(x_a) directly (the ground truth itself)

These baselines bound the achievable recovery error and let us interpret what the
trained model's macro_mse means relative to a source-blind ceiling.

Usage:
  python -m eval.synthetic_2x2.oracle_baselines --base_dir save/synthetic_2x2
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.synthetic_2x2.build_synth_dataset import (
    JOINTS_PER_SKEL, ACTIONS, T_FRAMES, K_SKELETONS, true_transport,
)


def evaluate_baseline(predict_fn, eval_pairs, J_max, T_frames):
    """For each (src_clip, skel_b, action) eval pair, compute MSE between
    predict_fn(src_clip, skel_b, action) and true_transport(src_clip, J_b, action).

    eval_pairs: list of (src_clip [T,J,3], skel_a, skel_b, action) tuples
    Returns dict of stats.
    """
    errors_by_action = {a: [] for a in range(len(ACTIONS))}
    for x_a, sa, sb, ai in eval_pairs:
        # CRITICAL: crop x_a to real joints before passing to true_transport.
        # Padded zero-joints would corrupt the chain interpolation.
        x_a_real = x_a[:, :JOINTS_PER_SKEL[sa], :]
        x_b_true = true_transport(x_a_real, JOINTS_PER_SKEL[sb], ACTIONS[ai])
        x_b_hat = predict_fn(x_a, sa, sb, ai, J_max, T_frames)  # [T, J_b, 3] or [T, J_max, 3]
        # Crop x_b_hat to actual joints
        if x_b_hat.shape[1] != JOINTS_PER_SKEL[sb]:
            x_b_hat = x_b_hat[:, :JOINTS_PER_SKEL[sb], :]
        err = float(np.mean((x_b_hat - x_b_true) ** 2))
        errors_by_action[ai].append(err)
    flat = [e for v in errors_by_action.values() for e in v]
    return {
        'macro_mse_per_action': {ACTIONS[a]: float(np.mean(v)) if v else None
                                  for a, v in errors_by_action.items()},
        'overall_macro_mse': float(np.mean([np.mean(v) for v in errors_by_action.values()
                                            if v])),
        'overall_mean_mse': float(np.mean(flat)),
        'overall_std_mse': float(np.std(flat)),
        'n_tested': len(flat),
    }


def predict_zero(x_a, sa, sb, ai, J_max, T):
    return np.zeros((T, JOINTS_PER_SKEL[sb], 3), dtype=np.float32)


def predict_random_gaussian(x_a, sa, sb, ai, J_max, T):
    return np.random.randn(T, JOINTS_PER_SKEL[sb], 3).astype(np.float32)


class CellMeanPredictor:
    """Source-blind conditional mean: E[T*(x_a) | skel_b, action]."""
    def __init__(self, training_clips, training_meta):
        # For each (skel_b, action) cell, compute mean of T*(x_a) over all training x_a's
        self.cache = {}
        # Group training clips by (action) — we'll compute T*(x_a) for each (skel_b, action) cell
        for sb in range(K_SKELETONS):
            for ai, action in enumerate(ACTIONS):
                # Find all training clips with action=ai (any skel_a != sb)
                src_indices = [i for i, m in enumerate(training_meta)
                               if m['action'] == ai and m['skel'] != sb]
                if not src_indices:
                    continue
                # Compute T*(x_a) for each training source, then average
                T_b_list = []
                for si in src_indices:
                    x_a = training_clips[si][:, :JOINTS_PER_SKEL[training_meta[si]['skel']], :]
                    x_b = true_transport(x_a, JOINTS_PER_SKEL[sb], action)
                    T_b_list.append(x_b)
                self.cache[(sb, ai)] = np.mean(np.stack(T_b_list), axis=0)

    def __call__(self, x_a, sa, sb, ai, J_max, T):
        return self.cache.get((sb, ai), predict_zero(x_a, sa, sb, ai, J_max, T))


class RandomSameActionPredictor:
    """Predict a random clip from the same (skel_b, action) cell of training data
    (after first applying T* to the chosen source). For sparse cells with one clip,
    this is the cell-mean predictor's source clip."""
    def __init__(self, training_clips, training_meta, seed=0):
        self.rng = np.random.RandomState(seed)
        # Index training clips by (skel_b, action)
        # Note: training clips are organized by (skel, action) — for predicting on skel_b,
        # we use a clip from the same (skel, action) cell as a stand-in. This is what an
        # ideal "average target clip" would predict.
        self.by_cell = {}
        for i, m in enumerate(training_meta):
            self.by_cell.setdefault((m['skel'], m['action']), []).append(i)
        self.training_clips = training_clips

    def __call__(self, x_a, sa, sb, ai, J_max, T):
        idxs = self.by_cell.get((sb, ai), [])
        if not idxs:
            return predict_zero(x_a, sa, sb, ai, J_max, T)
        i = idxs[self.rng.choice(len(idxs))]
        return self.training_clips[i][:, :JOINTS_PER_SKEL[sb], :]


def predict_oracle_with_xa(x_a, sa, sb, ai, J_max, T):
    """Direct T*(x_a) — perfect oracle (zero error by construction)."""
    x_a_real = x_a[:, :JOINTS_PER_SKEL[sa], :]
    return true_transport(x_a_real, JOINTS_PER_SKEL[sb], ACTIONS[ai])


def build_eval_pairs(meta, clips, n_pairs=200, seed=10):
    """Sample (src_clip, skel_a, skel_b, action) test triples, same protocol as
    eval_recovery in train_and_eval_synth.py.
    """
    rng = np.random.RandomState(seed)
    pairs = []
    for _ in range(n_pairs * 2):
        src = rng.choice(len(meta))
        sa = meta[src]['skel']
        ai = meta[src]['action']
        sb_choices = [k for k in range(K_SKELETONS) if k != sa]
        sb = int(sb_choices[rng.choice(len(sb_choices))])
        pairs.append((clips[src], sa, sb, ai))
        if len(pairs) >= n_pairs:
            break
    return pairs


def run_cell(cell_dir: Path, n_pairs=200, seed=0):
    print(f"\n=== Oracle baselines for cell: {cell_dir.name} ===")
    clips = np.load(cell_dir / 'clips.npy')
    meta = json.load(open(cell_dir / 'meta.json'))['clips']
    print(f"  {len(meta)} clips, J_max={clips.shape[2]}, T={clips.shape[1]}")

    # Re-derive train/eval split (same protocol as train_and_eval_synth.split_train_eval)
    rng = np.random.RandomState(seed)
    by_cell = {}
    for i, m in enumerate(meta):
        key = (m['skel'], m['action'])
        by_cell.setdefault(key, []).append(i)
    train_idx, eval_idx = [], []
    for key, idxs in by_cell.items():
        if len(idxs) == 1:
            train_idx.extend(idxs)
        else:
            shuffled = idxs.copy(); rng.shuffle(shuffled)
            n_eval = max(1, int(len(idxs) * 0.5))
            eval_idx.extend(shuffled[:n_eval])
            train_idx.extend(shuffled[n_eval:])
    if len(eval_idx) == 0:
        eval_idx = list(range(len(meta)))
    eval_meta = [meta[i] for i in eval_idx]
    eval_clips_list = [clips[i] for i in eval_idx]
    train_meta_list = [meta[i] for i in train_idx]
    train_clips_list = [clips[i] for i in train_idx]
    print(f"  train_idx={len(train_idx)}, eval_idx={len(eval_idx)}")

    eval_pairs = build_eval_pairs(eval_meta, eval_clips_list, n_pairs=n_pairs, seed=seed+10)
    print(f"  {len(eval_pairs)} eval pairs sampled")

    cell_mean = CellMeanPredictor(train_clips_list, train_meta_list)
    random_same = RandomSameActionPredictor(train_clips_list, train_meta_list, seed=seed)

    out = {}
    for name, fn in [
        ('zero', predict_zero),
        ('random_gaussian', predict_random_gaussian),
        ('random_same_action_clip', random_same),
        ('source_blind_cell_mean', cell_mean),
        ('oracle_with_xa', predict_oracle_with_xa),
    ]:
        np.random.seed(seed)
        m = evaluate_baseline(fn, eval_pairs, clips.shape[2], clips.shape[1])
        out[name] = m
        print(f"  {name:30s} macro_mse={m['overall_macro_mse']:.4f}")

    out_path = cell_dir / 'oracle_baselines.json'
    with open(out_path, 'w') as f:
        json.dump({'cell': cell_dir.name, 'n_train': len(train_idx),
                   'n_eval': len(eval_idx), 'n_pairs': len(eval_pairs),
                   'baselines': out}, f, indent=2)
    print(f"  Saved {out_path}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='save/synthetic_2x2')
    parser.add_argument('--n_pairs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    base_dir = PROJECT_ROOT / args.base_dir
    cells = ['sparse_unpaired', 'sparse_paired', 'dense_unpaired', 'dense_paired']

    summary = {}
    for c in cells:
        summary[c] = run_cell(base_dir / c, n_pairs=args.n_pairs, seed=args.seed)

    print(f"\n=== Synthetic 2x2 baseline comparison (overall_macro_mse) ===")
    print(f"{'cell':22s} {'zero':>8s} {'gauss':>8s} {'rand_same':>10s} {'cell_mean':>10s} {'oracle_xa':>10s}")
    for c in cells:
        b = summary[c]
        print(f"{c:22s} {b['zero']['overall_macro_mse']:8.4f} "
              f"{b['random_gaussian']['overall_macro_mse']:8.4f} "
              f"{b['random_same_action_clip']['overall_macro_mse']:10.4f} "
              f"{b['source_blind_cell_mean']['overall_macro_mse']:10.4f} "
              f"{b['oracle_with_xa']['overall_macro_mse']:10.4f}")

    out_summary = base_dir / 'oracle_baselines_summary.json'
    with open(out_summary, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved summary: {out_summary}")


if __name__ == '__main__':
    main()
