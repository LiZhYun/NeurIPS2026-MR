"""ANCHOR-Label-Flow-Src V5 inference: source-conditioned + label-conditioned.

Loads the trained AnchorLabelFlowSrcGenerator and produces per-query target-skeleton
motion arrays. The query gets:
  - cluster_id from I-5 RandomForest prediction on q_30d(source) (matches ANCHOR)
  - exact_id from V5 manifest's src_action label (matches ANCHOR)
  - src_z from per-skel VQ encoding of the source clip (NEW vs AL-Flow)
  - src_skel_id + src_graph from the source skeleton (NEW)
  - tgt_skel_id + tgt_graph from cond_dict (same as AL-Flow)

Sampling: greedy single-step Euler by default; multi-step Euler available.

Output:
  eval/results/baselines/anchor_label_flow_src_v5/fold_<seed>/
    query_NNNN.npy   ([T, J, 13] denormalized)
    metrics.json     (per-query status + counts)

Usage:
  python -m eval.baselines.run_anchor_label_flow_src_v5 \
      --ckpt save/actionbridge/anchor_label_flow_src_v1/ckpt_final.pt \
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
from model.actionbridge.generator_anchor_src import AnchorLabelFlowSrcGenerator
from model.moreflow.skel_graph import (
    SkelGraphEncoder, build_skel_features, pad_to_max_joints,
)
from model.moreflow.stage_a_registry import StageARegistry

DEFAULT_CKPT = 'save/actionbridge/anchor_label_flow_src_v1/ckpt_final.pt'

CLUSTERS_TR = sorted(ACTION_CLUSTERS.keys())
CLUSTER_TO_IDX = {c: i + 1 for i, c in enumerate(CLUSTERS_TR)}
EXACT_ACTIONS = sorted({a for acts in ACTION_CLUSTERS.values() for a in acts})
EXACT_ACTION_TO_IDX = {a: i + 1 for i, a in enumerate(EXACT_ACTIONS)}


def load_checkpoint(ckpt_path, device):
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = sd.get('args', {})
    skels = sd['skels']
    skel_to_id = sd['skel_to_id']
    n_clusters = sd['n_clusters']
    n_exact_actions = sd['n_exact_actions']

    G = AnchorLabelFlowSrcGenerator(
        n_skels=len(skels),
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


def sample_z_b(G, z_init, tgt_id, tgt_graph, src_z, src_id, src_graph,
               cluster_id, exact_id, n_steps=1, cfg_scale=1.0):
    """Greedy or multi-step Euler sampling with optional CFG."""
    z = z_init
    if n_steps == 1:
        t = torch.zeros(z.shape[0], device=z.device)
        v_cond = G(z, t, tgt_id, tgt_graph, src_z, src_id, src_graph,
                   cluster_id, exact_id)
        if cfg_scale != 1.0:
            cmask = torch.ones_like(cluster_id, dtype=torch.bool)
            emask = torch.ones_like(exact_id, dtype=torch.bool)
            smask = torch.ones_like(cluster_id, dtype=torch.bool)
            v_uncond = G(z, t, tgt_id, tgt_graph, src_z, src_id, src_graph,
                         cluster_id, exact_id,
                         cluster_mask=cmask, exact_mask=emask, src_mask=smask)
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v = v_cond
        return z + v
    dt = 1.0 / n_steps
    for k in range(n_steps):
        t = torch.full((z.shape[0],), k * dt, device=z.device)
        v_cond = G(z, t, tgt_id, tgt_graph, src_z, src_id, src_graph,
                   cluster_id, exact_id)
        if cfg_scale != 1.0:
            cmask = torch.ones_like(cluster_id, dtype=torch.bool)
            emask = torch.ones_like(exact_id, dtype=torch.bool)
            smask = torch.ones_like(cluster_id, dtype=torch.bool)
            v_uncond = G(z, t, tgt_id, tgt_graph, src_z, src_id, src_graph,
                         cluster_id, exact_id,
                         cluster_mask=cmask, exact_mask=emask, src_mask=smask)
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v = v_cond
        z = z + v * dt
    return z


CACHE_WINDOW = 32  # MUST match scripts/moreflow_extract_windows.py


def encode_source_via_stage_a(stage_a, skel_a, motion_phys):
    """Encode a [T, J, 13] physical motion into the per-skel VQ latent z [8, 256].

    Replicates the cache-build path exactly: pad/crop to CACHE_WINDOW=32 frames,
    normalize, nan_to_num, then encode_window. With downsample factor 4 the
    32-frame window produces an 8-token output that matches training cache rows.
    """
    if skel_a not in stage_a.tokenizers:
        return None
    motion_t = torch.as_tensor(motion_phys, dtype=torch.float32, device=stage_a.device)
    if motion_t.dim() == 3:
        motion_t = motion_t.unsqueeze(0)  # [1, T, J, 13]
    # Pad/crop to CACHE_WINDOW frames to match cache-build window length
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
        z, _ = stage_a.encode_window(skel_a, motion_norm)  # [1, T_tokens, codebook_dim]
    return z[0]  # [T_tokens, codebook_dim]


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
    parser.add_argument('--cfg_scale', type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    np.random.seed(args.seed); torch.manual_seed(args.seed)

    print(f"Loading ANCHOR-Label-Flow-Src ckpt: {args.ckpt}")
    G, skel_enc, skels, skel_to_id, ckpt_args = load_checkpoint(args.ckpt, device)
    print(f"  ckpt skels: {len(skels)}")

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
        out_dir = PROJECT_ROOT / f'eval/results/baselines/anchor_label_flow_src_v5/fold_{args.fold}'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    cond_dict = np.load(PROJECT_ROOT / 'dataset/truebones/zoo/truebones_processed/cond.npy',
                        allow_pickle=True).item()

    # Train I-5 classifier (same as ANCHOR uses, same as AL-Flow)
    print("Training I-5 RandomForest classifier on train_v3 Q cache...")
    qc_path = PROJECT_ROOT / 'idea-stage/quotient_cache.npz'
    qc = np.load(qc_path, allow_pickle=True)
    train_skels_set = set(OBJECT_SUBSETS_DICT['train_v3'])
    clf = train_classifier(qc, train_skels_set)
    qc_meta = qc['meta']
    fname_to_qc_idx = {m['fname']: i for i, m in enumerate(qc_meta)}
    I5_IDX_TO_CLUSTER = {v: k for k, v in I5_CLUSTER_TO_IDX.items()}
    print(f"  classifier ready; {len(fname_to_qc_idx)} qc rows")

    print("Building skel-graph features...")
    skel_features = build_skel_features(cond_dict)
    max_J = max(skel_features[s]['n_joints'] for s in skels if s in skel_features)
    pj_padded, pj_mask, pj_agg = {}, {}, {}
    for s in skels:
        if s not in skel_features:
            continue
        p, m = pad_to_max_joints(skel_features[s]['per_joint'], max_J)
        pj_padded[s] = p.to(device)
        pj_mask[s] = m.to(device)
        pj_agg[s] = skel_features[s]['agg'].to(device)

    # Stage A registry: need both source and target skels' tokenizers (encode + decode)
    needed_skels = sorted(set([q['skel_a'] for q in queries] + [q['skel_b'] for q in queries]))
    print(f"Loading Stage A tokenizers for {len(needed_skels)} skels (src+tgt)...")
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
            # Skip if skel_b not in trained set
            if skel_b not in skel_to_id:
                rec['status'] = 'skipped_skel_b_not_in_train'
                per_query.append(rec)
                continue
            if skel_b not in stage_a.tokenizers:
                rec['status'] = 'skipped_skel_b_no_stage_a'
                per_query.append(rec)
                continue
            if skel_a not in skel_to_id:
                rec['status'] = 'skipped_skel_a_not_in_train'
                per_query.append(rec)
                continue
            if skel_a not in stage_a.tokenizers:
                rec['status'] = 'skipped_skel_a_no_stage_a'
                per_query.append(rec)
                continue

            # Predict source cluster via I-5 RF
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

            # Exact action from manifest
            exact_id_int = EXACT_ACTION_TO_IDX.get(src_action, 0)

            # Load source motion + encode via Stage-A tokenizer (32-frame window, matches training cache)
            src_path = motion_dir / src_fname
            if not src_path.exists():
                rec['status'] = f'skipped_src_motion_missing: {src_fname}'
                per_query.append(rec)
                continue
            src_motion = np.load(src_path).astype(np.float32)
            src_z = encode_source_via_stage_a(stage_a, skel_a, src_motion)
            if src_z is None:
                rec['status'] = f'skipped_src_encode_failed'
                per_query.append(rec)
                continue
            if src_z.shape[0] != n_token:
                rec['status'] = f'skipped_src_encode_shape: got {tuple(src_z.shape)}'
                per_query.append(rec)
                continue

            # Sample target z (single batch)
            B = 1
            z_init = torch.randn(B, n_token, codebook_dim, device=device)
            tgt_id = torch.full((B,), skel_to_id[skel_b], dtype=torch.long, device=device)
            sid = torch.full((B,), skel_to_id[skel_a], dtype=torch.long, device=device)
            tgt_graph = skel_enc(pj_padded[skel_b].unsqueeze(0),
                                 pj_mask[skel_b].unsqueeze(0),
                                 pj_agg[skel_b].unsqueeze(0))
            src_graph = skel_enc(pj_padded[skel_a].unsqueeze(0),
                                 pj_mask[skel_a].unsqueeze(0),
                                 pj_agg[skel_a].unsqueeze(0))
            cid = torch.full((B,), cluster_id_int, dtype=torch.long, device=device)
            eid = torch.full((B,), exact_id_int, dtype=torch.long, device=device)
            src_z_b = src_z.unsqueeze(0)  # [1, 8, 256]

            with torch.no_grad():
                z_b = sample_z_b(G, z_init, tgt_id, tgt_graph,
                                 src_z_b, sid, src_graph, cid, eid,
                                 n_steps=args.n_euler_steps, cfg_scale=args.cfg_scale)

            # Decode through target Stage-A
            with torch.no_grad():
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
        'method': 'anchor_label_flow_src',
        'ckpt': str(args.ckpt),
        'fold': args.fold,
        'manifest': str(manifest_path),
        'n_queries': len(queries),
        'n_ok': sum(1 for r in per_query if r['status'] == 'ok'),
        'n_skipped': sum(1 for r in per_query if r['status'].startswith('skipped')),
        'n_error': sum(1 for r in per_query if r['status'].startswith('error')),
        'n_euler_steps': args.n_euler_steps,
        'cfg_scale': args.cfg_scale,
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
