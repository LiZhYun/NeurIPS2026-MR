"""Supervised CFM (paired) on v3 benchmark — Z-DTW only.

Per Codex audit (2026-04-22):
  - Existing ckpt is invariant-rep CFM trained on paired (src_inv, tgt_inv) for 100K steps
  - Same 60/10 train/test split as v3 benchmark
  - Output is invariant rep [T, 32, 8], NOT physical motion
  - Compatible with v3's Z-DTW metric only (skip pred→inv encode step)
  - Treat as "supervised invariant-space reference"

Pipeline:
  1. For each query: encode src motion → src_inv, run CFM Euler ODE → pred_inv [W, 32, 8]
  2. For each candidate (positive or adversarial): encode → candidate_inv
  3. Compute Z-DTW dist via dist_zscore_dtw_inv
  4. Per-query: AUC, contrastive_acc, closed_top1, best_match_dist
  5. Save fold-level summary JSON

Usage:
  python -m eval.baselines.run_supervised_cfm_v3 --folds 42 43
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.skel_blind.cfm_model import InvariantCFM, SLOT_COUNT, CHANNEL_COUNT
from model.skel_blind.encoder import encode_motion_to_invariant
from sample.generate_cfm import sample_euler
from eval.benchmark_v3.metrics_v3 import (
    dist_zscore_dtw_inv, contrastive_auc, contrastive_accuracy,
    best_match_distance, rank_of_best_positive, bootstrap_ci,
)

DATA_ROOT = PROJECT_ROOT / 'dataset/truebones/zoo/truebones_processed'
MOTION_DIR = DATA_ROOT / 'motions'
COND_PATH = DATA_ROOT / 'cond.npy'
CKPT_PATH = PROJECT_ROOT / 'save/cfm_supervised/ckpt_final.pt'
OUT_ROOT = PROJECT_ROOT / 'save/cfm_supervised/v3'
WINDOW = 40
N_ODE_STEPS = 50
CFG_WEIGHT = 2.0


def make_skel_cond(cond_dict, skel_name):
    return {
        'joints_names': cond_dict[skel_name]['joints_names'],
        'parents': cond_dict[skel_name]['parents'],
        'object_type': skel_name,
    }


def crop_or_pad(inv, window):
    T = inv.shape[0]
    if T >= window:
        return inv[:window]
    pad = np.zeros((window - T, SLOT_COUNT, CHANNEL_COUNT), dtype=np.float32)
    return np.concatenate([inv, pad], axis=0)


def evaluate_fold(model, device, cond_dict, fold_seed):
    manifest = json.load(open(PROJECT_ROOT / f'eval/benchmark_v3/queries/fold_{fold_seed}/manifest.json'))
    queries = manifest['queries']
    print(f"\n=== fold {fold_seed}: {len(queries)} queries ===")

    per_query_aucs = []
    per_query_cas = []
    per_query_ct1s = []
    per_query_bmds = []
    per_query_ranks = []
    per_split_aucs = {}
    per_split_cas = {}
    skipped = 0

    t0 = time.time()
    for i, q in enumerate(queries):
        skel_a = q['skel_a']
        skel_b = q['skel_b']
        if skel_a not in cond_dict or skel_b not in cond_dict:
            skipped += 1
            continue
        try:
            src_motion = np.load(MOTION_DIR / q['src_fname']).astype(np.float32)
            src_cond = make_skel_cond(cond_dict, skel_a)
            src_inv = encode_motion_to_invariant(src_motion, src_cond)
            src_inv_w = crop_or_pad(src_inv, WINDOW)

            # CFM inference
            with torch.no_grad():
                z_src = torch.from_numpy(src_inv_w).float().unsqueeze(0).to(device)
                out = sample_euler(model, z_src, n_steps=N_ODE_STEPS, cfg_weight=CFG_WEIGHT)
                pred_inv = out[0].cpu().numpy()

            # Reference invariant reps for positives + adversarials
            skel_b_cond = make_skel_cond(cond_dict, skel_b)
            pos_dists = []
            for p in q['positives']:
                ref_motion = np.load(MOTION_DIR / p['fname']).astype(np.float32)
                ref_inv = encode_motion_to_invariant(ref_motion, skel_b_cond)
                d = dist_zscore_dtw_inv(pred_inv, ref_inv)
                pos_dists.append(d)
            neg_dists = []
            for a in q['adversarials']:
                ref_motion = np.load(MOTION_DIR / a['fname']).astype(np.float32)
                ref_inv = encode_motion_to_invariant(ref_motion, skel_b_cond)
                d = dist_zscore_dtw_inv(pred_inv, ref_inv)
                neg_dists.append(d)

            auc = contrastive_auc(pos_dists, neg_dists)
            ca = contrastive_accuracy(pos_dists, neg_dists)
            ct1 = ca  # same as contrastive_accuracy in this protocol
            bmd = best_match_distance(pos_dists)
            rank = rank_of_best_positive(pos_dists, neg_dists)

            per_query_aucs.append(auc)
            per_query_cas.append(ca)
            per_query_ct1s.append(ct1)
            per_query_bmds.append(bmd)
            per_query_ranks.append(rank)
            per_split_aucs.setdefault(q['split'], []).append(auc)
            per_split_cas.setdefault(q['split'], []).append(ca)
        except Exception as e:
            print(f"  q{q['query_id']} ({skel_a}→{skel_b}): FAIL — {e}")
            skipped += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(queries) - i - 1)
            print(f"  [{i+1}/{len(queries)}] AUC≈{np.mean(per_query_aucs):.3f} "
                  f"CT1≈{np.mean(per_query_ct1s):.3f} ({elapsed:.0f}s, ETA {eta/60:.0f}min)")

    # Aggregate
    auc_arr = np.asarray(per_query_aucs)
    ct1_arr = np.asarray(per_query_ct1s)
    bmd_arr = np.asarray(per_query_bmds)
    auc_ci = bootstrap_ci(auc_arr)
    ct1_ci = bootstrap_ci(ct1_arr)
    bmd_ci = bootstrap_ci(bmd_arr)

    summary = {
        'method': 'cfm_supervised_invariant_zdtw',
        'fold': fold_seed,
        'n_queries': len(per_query_aucs),
        'n_skipped': skipped,
        'overall': {
            # bootstrap_ci returns (lo, mean, hi) — use full triple
            'auc': [auc_ci[0], float(auc_arr.mean()), auc_ci[2]],
            'closed_top_1': [ct1_ci[0], float(ct1_arr.mean()), ct1_ci[2]],
            'best_match_dist': [bmd_ci[0], float(bmd_arr.mean()), bmd_ci[2]],
        },
        'by_split': {
            s: {
                'n': len(v),
                'auc_mean': float(np.mean(v)),
                'ca_mean': float(np.mean(per_split_cas[s])),
            } for s, v in per_split_aucs.items()
        }
    }
    out_dir = OUT_ROOT / f'fold_{fold_seed}'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'v3_eval_zscore_dtw.json'
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {out_path}")
    print(f"  AUC = {auc_arr.mean():.3f} [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}]")
    print(f"  ClosedTop1 = {ct1_arr.mean():.3f} [{ct1_ci[0]:.3f}, {ct1_ci[1]:.3f}]")
    print(f"  BMD = {bmd_arr.mean():.3f}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', nargs='+', type=int, default=[42, 43])
    parser.add_argument('--ckpt', type=str, default=str(CKPT_PATH))
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print(f"Loading CFM ckpt: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cargs = ckpt['args']
    model = InvariantCFM(
        d_model=cargs.get('d_model', 384),
        n_layers=cargs.get('n_layers', 12),
        n_heads=cargs.get('n_heads', 8),
        max_frames=cargs.get('window', 40),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    cond_dict = np.load(COND_PATH, allow_pickle=True).item()
    print(f"Loaded cond: {len(cond_dict)} skels")

    for fold in args.folds:
        evaluate_fold(model, device, cond_dict, fold)


if __name__ == '__main__':
    main()
