"""I-5 baseline: Q-feature → action cluster classifier + class-conditioned retrieval.

Per Codex Round 1 brainstorm review (2026-04-23): same-day baseline gate.
Tells us how much pure action-label info gets us before training VQ-ActionBridge.

Pipeline:
  1. Featurize each Q clip → fixed 30-dim vector (mean+std per Q component + cadence)
  2. Train sklearn classifier on train_v3 60 skels (motion → cluster)
  3. For each v3 query: classify src → predicted cluster
  4. Retrieve any target_skel clip whose action_cluster matches predicted (random if multiple)
  5. Save as motion .npy
  6. Eval via standard v3 pipeline

Usage:
  python -m eval.baselines.run_i5_action_classifier_v3
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.benchmark_v3.action_taxonomy import (
    ACTION_CLUSTERS, parse_action_from_filename, action_to_cluster,
)
from data_loaders.truebones.truebones_utils.param_utils import OBJECT_SUBSETS_DICT

DATA_ROOT = PROJECT_ROOT / 'dataset/truebones/zoo/truebones_processed'
MOTION_DIR = DATA_ROOT / 'motions'
Q_CACHE_PATH = PROJECT_ROOT / 'idea-stage/quotient_cache.npz'
CLIP_INDEX_PATH = PROJECT_ROOT / 'eval/benchmark_v3/clip_index.json'
SAVE_ROOT = PROJECT_ROOT / 'save/oracles/v3/i5_action_classifier'
QUERIES_DIR_BY_VERSION = {
    'v3': PROJECT_ROOT / 'eval/benchmark_v3/queries',
    'v5': PROJECT_ROOT / 'eval/benchmark_v3/queries_v5',
}

CLUSTERS = sorted(ACTION_CLUSTERS.keys())
CLUSTER_TO_IDX = {c: i for i, c in enumerate(CLUSTERS)}


def featurize_q(com_path, heading_vel, contact_sched, cadence, limb_usage):
    """Convert one clip's Q to a fixed 30-dim feature vector."""
    feats = []
    # COM path: norm, mean step, std step, total length
    com = np.asarray(com_path, dtype=np.float32)
    if com.ndim == 1: com = com.reshape(-1, 3) if com.shape[0] % 3 == 0 else com.reshape(-1, 1)
    if com.shape[0] > 1:
        steps = np.linalg.norm(np.diff(com, axis=0), axis=-1)
        feats += [float(steps.mean()), float(steps.std()), float(steps.sum()),
                  float(np.linalg.norm(com[-1] - com[0]))]
    else:
        feats += [0.0, 0.0, 0.0, 0.0]
    # heading_vel: mean, std, total
    hv = np.asarray(heading_vel, dtype=np.float32)
    if hv.ndim == 1 and hv.size > 0:
        feats += [float(hv.mean()), float(hv.std()), float(np.abs(hv).mean())]
    elif hv.ndim == 2 and hv.size > 0:
        feats += [float(hv.mean()), float(hv.std()), float(np.abs(hv).mean())]
    else:
        feats += [0.0, 0.0, 0.0]
    # contact_sched: density per limb (top 6 dim by mean)
    cs = np.asarray(contact_sched, dtype=np.float32)
    if cs.ndim == 1: cs = cs.reshape(-1, 1)
    if cs.size > 0:
        density = cs.mean(axis=0)
        density_sorted = np.sort(density)[::-1]
        density_top6 = np.pad(density_sorted, (0, max(0, 6 - len(density_sorted))))[:6]
        feats += list(map(float, density_top6))
        # change rate
        if cs.shape[0] > 1:
            change_rate = float(np.abs(np.diff(cs.astype(np.float32), axis=0)).mean())
        else:
            change_rate = 0.0
        feats.append(change_rate)
    else:
        feats += [0.0] * 7
    # cadence
    feats.append(float(cadence))
    # limb_usage: top 6 sorted descending
    lu = np.asarray(limb_usage, dtype=np.float32)
    lu_sorted = np.sort(lu)[::-1]
    lu_top6 = np.pad(lu_sorted, (0, max(0, 6 - len(lu_sorted))))[:6]
    feats += list(map(float, lu_top6))
    # entropy of limb usage
    if lu.sum() > 0:
        p = lu / (lu.sum() + 1e-9)
        entropy = float(-(p * np.log(p + 1e-12)).sum())
    else:
        entropy = 0.0
    feats.append(entropy)
    return np.array(feats, dtype=np.float32)


def train_classifier(qc, train_skels):
    """Train classifier on train_skels' clips."""
    from sklearn.ensemble import RandomForestClassifier
    meta = qc['meta']
    X, y = [], []
    for i, m in enumerate(meta):
        if m['skeleton'] not in train_skels:
            continue
        # Use clip_index source of truth for action label
        action = parse_action_from_filename(m['fname'])
        cluster = action_to_cluster(action)
        if cluster is None or cluster not in CLUSTER_TO_IDX:
            continue
        feat = featurize_q(qc['com_path'][i], qc['heading_vel'][i],
                           qc['contact_sched'][i], qc['cadence'][i],
                           qc['limb_usage'][i])
        X.append(feat)
        y.append(CLUSTER_TO_IDX[cluster])
    X = np.array(X)
    y = np.array(y)
    print(f"Training set: {X.shape}, {len(set(y))} clusters")
    print(f"Cluster dist: {[(CLUSTERS[i], int((y==i).sum())) for i in range(len(CLUSTERS))]}")
    clf = RandomForestClassifier(n_estimators=200, max_depth=20, n_jobs=-1, random_state=42)
    clf.fit(X, y)
    train_acc = float((clf.predict(X) == y).mean())
    print(f"Train accuracy: {train_acc:.3f}")
    return clf


def build_target_pool():
    """{(skel, cluster) -> [fname, ...]}."""
    cidx = json.load(open(CLIP_INDEX_PATH))
    pool = defaultdict(list)
    for skel, clusters in cidx['index'].items():
        for cluster, clips in clusters.items():
            for clip in clips:
                pool[(skel, cluster)].append(clip['fname'])
    return pool


def gen_predictions_for_fold(clf, qc, target_pool, fold_seed, bench_version='v3'):
    manifest = json.load(open(QUERIES_DIR_BY_VERSION[bench_version] / f'fold_{fold_seed}/manifest.json'))
    rng = random.Random(fold_seed)
    suffix = '' if bench_version == 'v3' else f'_{bench_version}'
    out_dir = SAVE_ROOT.parent / f'i5_action_classifier{suffix}' / f'fold_{fold_seed}'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Map fname → q index for fast lookup
    fname_to_idx = {m['fname']: i for i, m in enumerate(qc['meta'])}

    n_done = 0
    n_correct = 0
    n_skipped = 0
    n_fallback = 0
    for q in manifest['queries']:
        qid = q['query_id']
        skel_b = q['skel_b']
        src_fname = q['src_fname']
        gt_action = q['src_action']
        gt_cluster = action_to_cluster(gt_action)
        # Featurize source
        if src_fname not in fname_to_idx:
            n_skipped += 1
            continue
        idx = fname_to_idx[src_fname]
        feat = featurize_q(qc['com_path'][idx], qc['heading_vel'][idx],
                           qc['contact_sched'][idx], qc['cadence'][idx],
                           qc['limb_usage'][idx])
        pred_cluster_idx = int(clf.predict(feat[None, :])[0])
        pred_cluster = CLUSTERS[pred_cluster_idx]
        if gt_cluster == pred_cluster:
            n_correct += 1
        # Retrieve target clip with predicted cluster
        cands = target_pool.get((skel_b, pred_cluster), [])
        if not cands:
            # Fallback: any clip on skel_b
            for c in CLUSTERS:
                cands = target_pool.get((skel_b, c), [])
                if cands: break
            n_fallback += 1
        if not cands:
            n_skipped += 1
            continue
        chosen = rng.choice(cands)
        motion = np.load(MOTION_DIR / chosen).astype(np.float32)
        np.save(out_dir / f'query_{qid:04d}.npy', motion)
        n_done += 1

    print(f"fold {fold_seed}: {n_done}/{len(manifest['queries'])} done, "
          f"{n_correct} correct ({100*n_correct/max(n_done,1):.1f}%), "
          f"{n_fallback} fallback, {n_skipped} skipped")
    return out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', nargs='+', type=int, default=[42, 43])
    parser.add_argument('--skip_eval', action='store_true')
    parser.add_argument('--bench_version', choices=['v3', 'v5'], default='v3')
    args = parser.parse_args()

    print("Loading Q cache...")
    qc = np.load(Q_CACHE_PATH, allow_pickle=True)
    print(f"  {len(qc['meta'])} clips, {len(set(m['skeleton'] for m in qc['meta']))} skels")

    train_skels = set(OBJECT_SUBSETS_DICT['train_v3'])
    print(f"Train skels: {len(train_skels)}")

    print("\nTraining classifier...")
    clf = train_classifier(qc, train_skels)

    print("\nBuilding target pool from clip_index...")
    target_pool = build_target_pool()
    print(f"  {len(target_pool)} (skel, cluster) entries")

    out_dirs = []
    for fold in args.folds:
        print(f"\n=== Generating predictions for fold {fold} (bench={args.bench_version}) ===")
        out_dirs.append(gen_predictions_for_fold(clf, qc, target_pool, fold, args.bench_version))

    if not args.skip_eval:
        import subprocess
        for fold, out_dir in zip(args.folds, out_dirs):
            for distance in ('procrustes', 'zscore_dtw', 'q_component'):
                cmd = [sys.executable, '-u', '-m', 'eval.benchmark_v3.eval_v3',
                       '--method_dir', str(out_dir),
                       '--fold', str(fold),
                       '--method_name', 'i5_action_classifier',
                       '--distance', distance]
                print(f"\n=== Eval: i5 fold {fold} {distance} ===")
                subprocess.run(cmd, cwd=str(PROJECT_ROOT))


if __name__ == '__main__':
    main()
