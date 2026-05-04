"""ANCHOR-Label-Flow-Src-Graph V5 inference: held-out-capable variant.

Forked from run_anchor_label_flow_src_v5.py with:
- Uses AnchorLabelFlowSrcGraphGenerator (no skel ID embeddings)
- DOES NOT skip held-out skel_a/skel_b (graph features handle ANY skeleton)
- Loads Stage A tokenizers for ALL needed skeletons (incl. test_v3)
- Computes skel_graph features for ALL skeletons via SkelGraphEncoder

This gives full V5 coverage (300 queries per fold including 100 test_test).

Usage:
  python -m eval.baselines.run_anchor_label_flow_src_g_v5 \
      --ckpt save/actionbridge/anchor_label_flow_src_g_v1/ckpt_final.pt \
      --fold 42 --max_queries 10000
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loaders.truebones.truebones_utils.param_utils import DATASET_DIR, OBJECT_SUBSETS_DICT
from eval.benchmark_v3.action_taxonomy import (
    ACTION_CLUSTERS, parse_action_from_filename, action_to_cluster,
)
from eval.baselines.run_i5_action_classifier_v3 import (
    train_classifier, CLUSTER_TO_IDX as I5_CLUSTER_TO_IDX, featurize_q,
)
from model.actionbridge.generator_anchor_src_g import AnchorLabelFlowSrcGraphGenerator
from model.moreflow.skel_graph import (
    SkelGraphEncoder, build_skel_features, pad_to_max_joints,
)
from model.moreflow.stage_a_registry import StageARegistry

DEFAULT_CKPT = 'save/actionbridge/anchor_label_flow_src_g_v1/ckpt_final.pt'

CLUSTERS_TR = sorted(ACTION_CLUSTERS.keys())
CLUSTER_TO_IDX = {c: i + 1 for i, c in enumerate(CLUSTERS_TR)}
EXACT_ACTIONS = sorted({a for acts in ACTION_CLUSTERS.values() for a in acts})
EXACT_ACTION_TO_IDX = {a: i + 1 for i, a in enumerate(EXACT_ACTIONS)}

CACHE_WINDOW = 32  # MUST match scripts/moreflow_extract_windows.py


def load_checkpoint(ckpt_path, device):
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = sd.get('args', {})
    skels = sd['skels']
    skel_to_id = sd['skel_to_id']
    n_clusters = sd['n_clusters']
    n_exact_actions = sd['n_exact_actions']

    G = AnchorLabelFlowSrcGraphGenerator(
        n_clusters=n_clusters,
        n_exact_actions=n_exact_actions,
        codebook_dim=256,
        d_model=args.get('d_model', 512),
        n_layers=args.get('n_layers', 6),
        n_heads=args.get('n_heads', 8),
    ).to(device)
    G.load_state_dict(sd['G'])
    G.eval()

    skel_enc = SkelGraphEncoder().to(device)
    skel_enc.load_state_dict(sd['skel_enc'])
    skel_enc.eval()

    return G, skel_enc, skels, skel_to_id, args


def encode_source_via_stage_a(stage_a, skel_a, motion_phys):
    if skel_a not in stage_a.tokenizers:
        return None
    motion_t = torch.as_tensor(motion_phys, dtype=torch.float32, device=stage_a.device)
    if motion_t.dim() == 3:
        motion_t = motion_t.unsqueeze(0)
    T = motion_t.shape[1]
    if T < CACHE_WINDOW:
        pad_T = CACHE_WINDOW - T
        last = motion_t[:, -1:, :, :]
        pad = last.expand(-1, pad_T, -1, -1)
        motion_t = torch.cat([motion_t, pad], dim=1)
    elif T > CACHE_WINDOW:
        motion_t = motion_t[:, :CACHE_WINDOW, :, :]
    with torch.no_grad():
        motion_norm = stage_a.normalize(skel_a, motion_t)
        motion_norm = torch.nan_to_num(motion_norm)
        z, _ = stage_a.encode_window(skel_a, motion_norm)
    return z[0]


def sample_z_b(G, z_init, tgt_graph, src_z, src_graph,
               cluster_id, exact_id, n_steps=1):
    """Greedy or multi-step Euler sampling."""
    z = z_init
    if n_steps == 1:
        t = torch.zeros(z.shape[0], device=z.device)
        v = G(z, t, tgt_graph, src_z, src_graph, cluster_id, exact_id)
        return z + v
    dt = 1.0 / n_steps
    for k in range(n_steps):
        t = torch.full((z.shape[0],), k * dt, device=z.device)
        v = G(z, t, tgt_graph, src_z, src_graph, cluster_id, exact_id)
        z = z + v * dt
    return z


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default=DEFAULT_CKPT)
    parser.add_argument('--fold', type=int, default=42)
    parser.add_argument('--manifest', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--max_queries', type=int, default=10000)
    parser.add_argument('--motion_dir', type=str,
                        default='dataset/truebones/zoo/truebones_processed/motions')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_euler_steps', type=int, default=1)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    print(f"Loading AL-Flow-Src-G ckpt: {args.ckpt}")
    G, skel_enc, train_skels_list, skel_to_id, ckpt_args = load_checkpoint(args.ckpt, device)
    print(f"  ckpt train_skels: {len(train_skels_list)}")

    if args.manifest:
        manifest_path = args.manifest
    else:
        manifest_path = PROJECT_ROOT / f'eval/benchmark_v3/queries_v5/fold_{args.fold}/manifest.json'
    with open(manifest_path) as f:
        manifest = json.load(f)
    queries = manifest['queries'][:args.max_queries]
    print(f"V5 fold {args.fold}: {len(queries)} queries")

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = PROJECT_ROOT / f'eval/results/baselines/anchor_label_flow_src_g_v5/fold_{args.fold}'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    cond_dict = np.load(PROJECT_ROOT / 'dataset/truebones/zoo/truebones_processed/cond.npy',
                        allow_pickle=True).item()

    print("Training I-5 RandomForest classifier on train_v3 Q cache...")
    qc_path = PROJECT_ROOT / 'idea-stage/quotient_cache.npz'
    qc = np.load(qc_path, allow_pickle=True)
    train_skels_set = set(OBJECT_SUBSETS_DICT['train_v3'])
    clf = train_classifier(qc, train_skels_set)
    qc_meta = qc['meta']
    fname_to_qc_idx = {m['fname']: i for i, m in enumerate(qc_meta)}
    I5_IDX_TO_CLUSTER = {v: k for k, v in I5_CLUSTER_TO_IDX.items()}
    print(f"  classifier ready; {len(fname_to_qc_idx)} qc rows")

    # Build skel-graph features for ALL skeletons (incl. held-out test_v3)
    print("Building skel-graph features for ALL skeletons (train + test_v3)...")
    skel_features = build_skel_features(cond_dict)
    all_skels_in_queries = sorted(set([q['skel_a'] for q in queries] + [q['skel_b'] for q in queries]))
    available = [s for s in all_skels_in_queries if s in skel_features]
    print(f"  available: {len(available)}/{len(all_skels_in_queries)}")
    max_J = max(skel_features[s]['n_joints'] for s in available)
    pj_padded, pj_mask, pj_agg = {}, {}, {}
    for s in available:
        p, m = pad_to_max_joints(skel_features[s]['per_joint'], max_J)
        pj_padded[s] = p.to(device)
        pj_mask[s] = m.to(device)
        pj_agg[s] = skel_features[s]['agg'].to(device)

    # Stage A registry for both source and target skeletons (full V5 coverage)
    needed_skels = available
    print(f"Loading Stage A tokenizers for {len(needed_skels)} skels...")
    stage_a = StageARegistry(needed_skels, device=str(device))

    per_query = []
    n_token = 8
    codebook_dim = 256
    t_total_0 = time.time()
    motion_dir = PROJECT_ROOT / args.motion_dir

    for i, q in enumerate(queries):
        qid = q['query_id']
        skel_a = q.get('skel_a', '')
        skel_b = q['skel_b']
        src_action = q.get('src_action', '')
        cluster_gt = q.get('cluster', '')
        split = q.get('split', '')

        rec = {'query_id': qid, 'cluster': cluster_gt, 'split': split,
               'skel_a': skel_a, 'skel_b': skel_b,
               'src_action': src_action, 'status': 'pending'}

        try:
            # NO skel_id fallback — but we need graph features. Skip if graph missing.
            if skel_b not in pj_padded:
                rec['status'] = 'skipped_skel_b_no_graph'
                per_query.append(rec)
                continue
            if skel_b not in stage_a.tokenizers:
                rec['status'] = 'skipped_skel_b_no_stage_a'
                per_query.append(rec)
                continue
            if skel_a not in pj_padded:
                rec['status'] = 'skipped_skel_a_no_graph'
                per_query.append(rec)
                continue
            if skel_a not in stage_a.tokenizers:
                rec['status'] = 'skipped_skel_a_no_stage_a'
                per_query.append(rec)
                continue

            # Predict source cluster (same as ANCHOR + AL-Flow)
            src_fname = q['src_fname']
            if src_fname in fname_to_qc_idx:
                qi = fname_to_qc_idx[src_fname]
                feat = featurize_q(qc['com_path'][qi], qc['heading_vel'][qi],
                                   qc['contact_sched'][qi], qc['cadence'][qi],
                                   qc['limb_usage'][qi])
                pred_cluster_idx = int(clf.predict(feat.reshape(1, -1))[0])
                pred_cluster_str = I5_IDX_TO_CLUSTER.get(pred_cluster_idx, '')
            else:
                pred_cluster_str = ''
            cluster_id_int = CLUSTER_TO_IDX.get(pred_cluster_str, 0)
            exact_id_int = EXACT_ACTION_TO_IDX.get(src_action, 0)

            # Encode source motion
            src_path = motion_dir / src_fname
            if not src_path.exists():
                rec['status'] = f'skipped_src_motion_missing'
                per_query.append(rec)
                continue
            src_motion = np.load(src_path).astype(np.float32)
            src_z = encode_source_via_stage_a(stage_a, skel_a, src_motion)
            if src_z is None or src_z.shape[0] != n_token:
                rec['status'] = f'skipped_src_encode_failed'
                per_query.append(rec)
                continue

            # Compute graph features for src + tgt (handles ANY skeleton)
            B = 1
            tgt_graph = skel_enc(pj_padded[skel_b].unsqueeze(0),
                                 pj_mask[skel_b].unsqueeze(0),
                                 pj_agg[skel_b].unsqueeze(0))
            src_graph = skel_enc(pj_padded[skel_a].unsqueeze(0),
                                 pj_mask[skel_a].unsqueeze(0),
                                 pj_agg[skel_a].unsqueeze(0))

            cid = torch.full((B,), cluster_id_int, dtype=torch.long, device=device)
            eid = torch.full((B,), exact_id_int, dtype=torch.long, device=device)
            src_z_b = src_z.unsqueeze(0)
            z_init = torch.randn(B, n_token, codebook_dim, device=device)

            with torch.no_grad():
                z_b = sample_z_b(G, z_init, tgt_graph, src_z_b, src_graph,
                                 cid, eid, n_steps=args.n_euler_steps)
                motion_norm = stage_a.decode_tokens(skel_b, z_b)
                motion_phys = stage_a.unnormalize(skel_b, motion_norm)

            sample = motion_phys[0].cpu().numpy().astype(np.float32)
            pos_T = [p['T'] for p in q.get('positives_cluster', [])]
            T_tgt = int(np.median(pos_T)) if pos_T else q.get('src_T', sample.shape[0])
            T_out = min(T_tgt, sample.shape[0])
            sample = sample[:T_out]

            np.save(out_dir / f'query_{qid:04d}.npy', sample)
            rec['status'] = 'ok'
            rec['T_out'] = int(sample.shape[0])
            rec['pred_cluster'] = pred_cluster_str
            rec['exact_id'] = int(exact_id_int)

        except Exception as e:
            rec['status'] = f'error: {type(e).__name__}: {e}'
            rec['traceback'] = traceback.format_exc()

        per_query.append(rec)

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t_total_0
            rate = (i + 1) / max(1, elapsed)
            eta = (len(queries) - i - 1) / max(1e-6, rate)
            ok = sum(1 for r in per_query if r['status'] == 'ok')
            print(f"  [{i+1:4d}/{len(queries)}] ok={ok} elapsed={elapsed:.0f}s rate={rate:.1f}/s ETA={eta:.0f}s")

    summary = {
        'method': 'anchor_label_flow_src_g',
        'ckpt': str(args.ckpt),
        'fold': args.fold,
        'manifest': str(manifest_path),
        'n_queries': len(queries),
        'n_ok': sum(1 for r in per_query if r['status'] == 'ok'),
        'n_skipped': sum(1 for r in per_query if r['status'].startswith('skipped')),
        'n_error': sum(1 for r in per_query if r['status'].startswith('error')),
        'n_euler_steps': args.n_euler_steps,
        'wall_clock_s': time.time() - t_total_0,
        'per_query': per_query,
    }
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_dir}/metrics.json")
    print(f"  ok={summary['n_ok']} / skipped={summary['n_skipped']} / error={summary['n_error']}")
    print(f"  wall_clock={summary['wall_clock_s']:.0f}s")


if __name__ == '__main__':
    main()
