"""VQ-ActionBridge inference: action-conditioned generation on per-skel VQ.

Pipeline:
  1. Classify src motion → action_cluster (using I-5 RandomForest on Q-features)
  2. Sample target z from G(noise, action_cluster, skel_b) via Euler ODE
  3. Decode via frozen MoReFlow Stage A per-skel VQ-VAE
  4. Save as physical motion .npy

Usage:
  python -m sample.actionbridge_retarget --ckpt save/actionbridge/actionbridge_v1/ckpt_final.pt --fold 42
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loaders.truebones.truebones_utils.param_utils import OBJECT_SUBSETS_DICT
from eval.benchmark_v3.action_taxonomy import (
    ACTION_CLUSTERS, parse_action_from_filename, action_to_cluster,
)
from model.actionbridge.generator import ActionBridgeGenerator
from model.moreflow.skel_graph import (
    SkelGraphEncoder, build_skel_features, pad_to_max_joints,
)
from model.moreflow.stage_a_registry import StageARegistry

DATA_ROOT = PROJECT_ROOT / 'dataset/truebones/zoo/truebones_processed'
COND_PATH = DATA_ROOT / 'cond.npy'
MOTION_DIR = DATA_ROOT / 'motions'
Q_CACHE_PATH = PROJECT_ROOT / 'idea-stage/quotient_cache.npz'

CLUSTERS = sorted(ACTION_CLUSTERS.keys())
ACTION_TO_IDX = {c: i + 1 for i, c in enumerate(CLUSTERS)}
N_ACTIONS = len(CLUSTERS) + 1


def load_classifier():
    """Load Q-feature → cluster RandomForest classifier (from I-5 baseline)."""
    # Re-import I-5 train code
    from eval.baselines.run_i5_action_classifier_v3 import (
        train_classifier, featurize_q
    )
    qc = np.load(Q_CACHE_PATH, allow_pickle=True)
    train_skels = set(OBJECT_SUBSETS_DICT['train_v3'])
    print("Training I-5 classifier (for action prediction)...")
    clf = train_classifier(qc, train_skels)
    fname_to_idx = {m['fname']: i for i, m in enumerate(qc['meta'])}
    return clf, qc, fname_to_idx, featurize_q


@torch.no_grad()
def sample_z(G, action_id, tgt_skel_id, tgt_graph, device, n_steps=50, cfg_w=2.0):
    """Euler ODE sampling from noise to target z [1, 8, 256]."""
    B = action_id.shape[0]
    x = torch.randn(B, 8, 256, device=device)
    null_id = torch.zeros_like(action_id)  # 0 = NULL action
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((B,), i * dt, device=device)
        v_cond = G(x, t, tgt_skel_id, tgt_graph, action_id)
        if cfg_w > 0:
            v_uncond = G(x, t, tgt_skel_id, tgt_graph, null_id)
            v = (1 + cfg_w) * v_cond - cfg_w * v_uncond
        else:
            v = v_cond
        x = x + v * dt
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--fold', type=int, default=42)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--n_steps', type=int, default=50)
    parser.add_argument('--cfg_w', type=float, default=2.0)
    parser.add_argument('--n_samples', type=int, default=1, help='samples per query (rerank)')
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    print(f"Loading ckpt: {args.ckpt}")
    sd = torch.load(args.ckpt, map_location=device, weights_only=False)
    skels = sd['skels']
    skel_to_id = sd['skel_to_id']
    n_actions = sd.get('n_actions', N_ACTIONS)
    # Load training args for arch settings (d_model, n_layers)
    train_args_path = Path(args.ckpt).parent / 'args.json'
    train_args = json.load(open(train_args_path)) if train_args_path.exists() else {}
    d_model = train_args.get('d_model', 384)
    n_layers = train_args.get('n_layers', 6)
    print(f"Model arch: d_model={d_model}, n_layers={n_layers}, n_actions={n_actions}")
    G = ActionBridgeGenerator(n_skels=len(skels), n_actions=n_actions,
                              d_model=d_model, n_layers=n_layers).to(device)
    G.load_state_dict(sd['G'])
    G.eval()
    skel_enc = SkelGraphEncoder().to(device)
    skel_enc.load_state_dict(sd['skel_enc'])
    skel_enc.eval()
    print(f"Trained on {len(skels)} skels")

    # Load Stage A registry (for decoding) — need ALL 70 skels for inference (incl. test_test)
    print("Loading Stage A registry for ALL 70 skels...")
    all_skels = list(set(OBJECT_SUBSETS_DICT['train_v3']) | set(OBJECT_SUBSETS_DICT['test_v3']))
    registry = StageARegistry(all_skels, device=device)

    # Skel graph features
    cond_dict = np.load(COND_PATH, allow_pickle=True).item()
    skel_features = build_skel_features(cond_dict)
    max_J = max(skel_features[s]['n_joints'] for s in all_skels)
    pj_padded, pj_mask, pj_agg = {}, {}, {}
    for s in all_skels:
        p, m = pad_to_max_joints(skel_features[s]['per_joint'], max_J)
        pj_padded[s] = p.to(device)
        pj_mask[s] = m.to(device)
        pj_agg[s] = skel_features[s]['agg'].to(device)

    # Action classifier (from I-5)
    clf, qc, fname_to_idx, featurize_q = load_classifier()

    # Load query manifest
    manifest = json.load(open(PROJECT_ROOT / f'eval/benchmark_v3/queries/fold_{args.fold}/manifest.json'))
    queries = manifest['queries']
    if args.limit:
        queries = queries[:args.limit]
    print(f"Processing {len(queries)} queries from fold {args.fold}...")

    out_dir = Path(args.out_dir) if args.out_dir else (
        PROJECT_ROOT / f'save/actionbridge_inference/v3/fold_{args.fold}')
    out_dir.mkdir(parents=True, exist_ok=True)

    n_done = n_skipped = n_unk_skel = 0
    t0 = time.time()
    for i, q in enumerate(queries):
        qid = q['query_id']
        out_path = out_dir / f'query_{qid:04d}.npy'
        if out_path.exists():
            n_done += 1
            continue
        try:
            skel_b = q['skel_b']
            src_fname = q['src_fname']
            # Predict action
            if src_fname not in fname_to_idx:
                n_skipped += 1
                continue
            idx = fname_to_idx[src_fname]
            feat = featurize_q(qc['com_path'][idx], qc['heading_vel'][idx],
                               qc['contact_sched'][idx], qc['cadence'][idx],
                               qc['limb_usage'][idx])
            pred_cluster_idx = int(clf.predict(feat[None, :])[0])
            pred_cluster = sorted(ACTION_CLUSTERS.keys())[pred_cluster_idx]
            action_id = ACTION_TO_IDX[pred_cluster]

            # Sample target z
            if skel_b not in skel_to_id:
                # Fallback: use target skel features only, action conditioning, but skel_id=0
                tid = 0
                n_unk_skel += 1
            else:
                tid = skel_to_id[skel_b]
            tid_t = torch.tensor([tid], dtype=torch.long, device=device)
            tg = skel_enc(pj_padded[skel_b].unsqueeze(0), pj_mask[skel_b].unsqueeze(0),
                          pj_agg[skel_b].unsqueeze(0))
            aid_t = torch.tensor([action_id], dtype=torch.long, device=device)

            # Single sample (could be multi-sample + rerank later)
            z_pred = sample_z(G, aid_t, tid_t, tg, device,
                              n_steps=args.n_steps, cfg_w=args.cfg_w)  # [1, 8, 256]

            # Decode via per-skel VQ
            x_norm = registry.decode_tokens(skel_b, z_pred)
            x_phys = registry.unnormalize(skel_b, x_norm).squeeze(0).cpu().numpy().astype(np.float32)
            np.save(out_path, x_phys)
            n_done += 1
        except Exception as e:
            print(f"  q{qid} ({q['skel_a']}→{q['skel_b']}): FAIL — {e}")
            import traceback; traceback.print_exc()
            n_skipped += 1
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(queries) - i - 1)
            print(f"  [{i+1}/{len(queries)}] done={n_done} skipped={n_skipped} unk_skel={n_unk_skel} "
                  f"({elapsed:.0f}s, ETA {eta/60:.0f}min)")

    print(f"\nFinished: {n_done} written, {n_skipped} skipped, {n_unk_skel} unk-skel-fallback")


if __name__ == '__main__':
    main()
