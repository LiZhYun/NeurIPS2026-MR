"""Train ANCHOR-Label-Flow: cluster + exact-action conditional flow on per-skel VQ latents.

This is the supervision-matched generative comparator to ANCHOR (the retrieval method).
ANCHOR uses both the coarse action-cluster and fine exact-action labels from Truebones
filenames; this generator gets the same two label channels via FiLM conditioning. If it
still loses to ANCHOR on V5 cluster-tier AUC, the failure is structural (Theorem 1 /
Proposition 1), not metadata-availability.

Forked from train/train_actionbridge.py with:
- exact-action label extracted alongside cluster
- AnchorLabelFlowGenerator (model/actionbridge/generator_anchor.py) replaces
  ActionBridgeGenerator
- dual CFG dropout (cluster_drop and exact_drop independent, defaults 10% each)
- checkpoint stores both n_clusters and n_exact_actions

Usage:
  python -m train.train_anchor_label_flow --run_name anchor_label_flow_v1 --max_steps 50000
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loaders.truebones.truebones_utils.param_utils import OBJECT_SUBSETS_DICT
from eval.benchmark_v3.action_taxonomy import (
    ACTION_CLUSTERS, parse_action_from_filename, action_to_cluster,
)
from model.actionbridge.generator_anchor import (
    AnchorLabelFlowGenerator, count_parameters,
)
from model.moreflow.skel_graph import (
    SkelGraphEncoder, build_skel_features, pad_to_max_joints,
)

DATA_ROOT = PROJECT_ROOT / 'dataset/truebones/zoo/truebones_processed'
COND_PATH = DATA_ROOT / 'cond.npy'
CACHE_PATH = PROJECT_ROOT / 'save/moreflow_flow/cache_train_v3.pt'
SAVE_ROOT = PROJECT_ROOT / 'save/actionbridge'

# Cluster vocab: index 0 = NULL (CFG drop), 1..N = real clusters
CLUSTERS = sorted(ACTION_CLUSTERS.keys())
CLUSTER_TO_IDX = {c: i + 1 for i, c in enumerate(CLUSTERS)}
N_CLUSTERS = len(CLUSTERS) + 1  # +1 for NULL

# Exact-action vocab: index 0 = NULL (CFG drop), 1..N = real exact actions
EXACT_ACTIONS = sorted({a for acts in ACTION_CLUSTERS.values() for a in acts})
EXACT_ACTION_TO_IDX = {a: i + 1 for i, a in enumerate(EXACT_ACTIONS)}
N_EXACT_ACTIONS = len(EXACT_ACTIONS) + 1  # +1 for NULL


def lr_lambda(step, warmup, total):
    if warmup > 0 and step < warmup:
        return float(step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def build_training_data(cache, skel_to_id):
    """Build per-row mapping → (skel_name, row_idx, cluster_idx, exact_idx).

    Each row carries BOTH coarse cluster and fine exact-action labels.
    Rows whose action is unmappable to a cluster OR not in EXACT_ACTIONS are skipped.
    """
    rows = []
    skipped_cluster = 0
    skipped_exact = 0
    for skel_name, data in cache.items():
        if skel_name.startswith('_'):  # skip _meta and other reserved keys
            continue
        meta = data['meta']  # list of (fname, frame_offset)
        for ri, (fname, _) in enumerate(meta):
            action = parse_action_from_filename(fname)
            cluster = action_to_cluster(action)
            if cluster is None:
                skipped_cluster += 1
                continue
            if action not in EXACT_ACTION_TO_IDX:
                skipped_exact += 1
                continue
            cid = CLUSTER_TO_IDX[cluster]
            eid = EXACT_ACTION_TO_IDX[action]
            rows.append((skel_name, ri, cid, eid))
    print(f"Built {len(rows)} training rows "
          f"(skipped {skipped_cluster} unmappable cluster, "
          f"{skipped_exact} unmappable exact-action)")
    return rows


def train(args):
    save_dir = SAVE_ROOT / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / 'training_log.jsonl'

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}, run_name: {args.run_name}")
    print(f"Vocab: n_clusters={N_CLUSTERS} (10 + NULL), n_exact_actions={N_EXACT_ACTIONS} (124 + NULL)")

    print(f"Loading cache: {CACHE_PATH}")
    cache = torch.load(CACHE_PATH, map_location='cpu', weights_only=False)
    skels = sorted([s for s in cache.keys() if not s.startswith('_')])
    skel_to_id = {s: i for i, s in enumerate(skels)}
    print(f"Cache: {len(skels)} skels")

    rows = build_training_data(cache, skel_to_id)
    if not rows:
        raise RuntimeError("No training rows after filtering — check action taxonomy.")

    # Pin caches to GPU as fp32
    z_per_skel = {s: cache[s]['z_continuous'].float().to(device) for s in skels}
    n_per_skel = {s: z_per_skel[s].shape[0] for s in skels}

    # Skel graph features
    print("Building skel-graph features...")
    cond_dict = np.load(COND_PATH, allow_pickle=True).item()
    skel_features = build_skel_features(cond_dict)
    max_J = max(skel_features[s]['n_joints'] for s in skels)
    pj_padded, pj_mask, pj_agg = {}, {}, {}
    for s in skels:
        p, m = pad_to_max_joints(skel_features[s]['per_joint'], max_J)
        pj_padded[s] = p.to(device)
        pj_mask[s] = m.to(device)
        pj_agg[s] = skel_features[s]['agg'].to(device)
    skel_enc = SkelGraphEncoder().to(device)

    # Generator (dual cluster + exact-action conditioning)
    G = AnchorLabelFlowGenerator(
        n_skels=len(skels),
        n_clusters=N_CLUSTERS,
        n_exact_actions=N_EXACT_ACTIONS,
        codebook_dim=256,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
    ).to(device)
    n_g = count_parameters(G)
    n_se = count_parameters(skel_enc)
    print(f"Generator params: {n_g/1e6:.1f}M, skel_enc: {n_se/1e6:.1f}M")

    optim = torch.optim.AdamW(list(G.parameters()) + list(skel_enc.parameters()),
                              lr=args.lr, weight_decay=0.01, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.LambdaLR(optim,
        lambda s: lr_lambda(s, args.warmup, args.max_steps))

    # Save args + vocab to disk for resume + eval
    with open(save_dir / 'args.json', 'w') as f:
        json.dump({**vars(args), 'n_skels': len(skels), 'skels': skels,
                   'n_clusters': N_CLUSTERS, 'clusters': CLUSTERS,
                   'n_exact_actions': N_EXACT_ACTIONS, 'exact_actions': EXACT_ACTIONS,
                   'max_J': max_J}, f, indent=2)

    # Resume
    rng = np.random.RandomState(args.seed + 1)
    start_step = 0
    ckpts = sorted(save_dir.glob('ckpt_step*.pt'))
    if ckpts and not args.no_resume:
        latest = ckpts[-1]
        print(f"Resuming from {latest.name}")
        sd = torch.load(latest, map_location=device, weights_only=False)
        G.load_state_dict(sd['G'])
        skel_enc.load_state_dict(sd['skel_enc'])
        optim.load_state_dict(sd['optim'])
        start_step = sd['step']
        rng.set_state(sd['rng'])

    print(f"Training {args.max_steps - start_step} steps from {start_step}")
    t0 = time.time()
    G.train(); skel_enc.train()
    losses_window = []
    for step in range(start_step, args.max_steps):
        optim.zero_grad()
        # Sample batch
        idx = rng.choice(len(rows), args.batch_size)
        batch_rows = [rows[i] for i in idx]
        # Group by skel for efficient stacking (single skel per inner loop iter)
        skel_groups = defaultdict(list)
        for (s, ri, cid, eid) in batch_rows:
            skel_groups[s].append((ri, cid, eid))

        total_loss = 0.0
        n_micro = 0
        for skel, items in skel_groups.items():
            ris = torch.tensor([it[0] for it in items], dtype=torch.long, device=device)
            cids = torch.tensor([it[1] for it in items], dtype=torch.long, device=device)
            eids = torch.tensor([it[2] for it in items], dtype=torch.long, device=device)
            B = len(items)

            z1 = z_per_skel[skel][ris]                      # [B, 8, 256] target
            z0 = torch.randn_like(z1)                       # noise
            q = torch.rand(B, device=device)                # flow time
            zt = (1 - q.view(-1, 1, 1)) * z0 + q.view(-1, 1, 1) * z1
            v_target = z1 - z0                              # straight-line interpolant velocity

            # Independent CFG dropouts: cluster and exact-action drop separately
            cluster_drop = (torch.rand(B, device=device) < args.p_cluster_drop)
            exact_drop = (torch.rand(B, device=device) < args.p_exact_drop)

            tid = torch.full((B,), skel_to_id[skel], dtype=torch.long, device=device)
            tg = skel_enc(pj_padded[skel].unsqueeze(0), pj_mask[skel].unsqueeze(0),
                          pj_agg[skel].unsqueeze(0)).expand(B, -1)

            v_pred = G(zt, q, tid, tg, cids, eids,
                       cluster_mask=cluster_drop, exact_mask=exact_drop)
            loss = F.mse_loss(v_pred, v_target)
            loss.backward()
            total_loss += loss.item()
            n_micro += 1

        if n_micro == 0:
            continue
        torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(skel_enc.parameters(), 1.0)
        optim.step()
        sched.step()
        avg_loss = total_loss / n_micro
        losses_window.append(avg_loss)
        if len(losses_window) > 200: losses_window = losses_window[-200:]

        if (step + 1) % args.log_interval == 0:
            elapsed = time.time() - t0
            eta = elapsed / max(1, step + 1 - start_step) * (args.max_steps - step - 1)
            ravg = np.mean(losses_window)
            cur_lr = sched.get_last_lr()[0]
            print(f"  [{step+1:6d}/{args.max_steps}] loss={avg_loss:.4f} avg200={ravg:.4f} "
                  f"lr={cur_lr:.2e} ({elapsed:.0f}s, ETA {eta/60:.0f}min)")
            with open(log_path, 'a') as f:
                f.write(json.dumps({'step': step + 1, 'loss': avg_loss,
                                    'avg200': float(ravg), 'lr': cur_lr}) + '\n')

        if (step + 1) % args.ckpt_interval == 0 and (step + 1) < args.max_steps:
            ck = save_dir / f'ckpt_step{step+1:07d}.pt'
            tmp = ck.with_suffix('.pt.tmp')
            torch.save({
                'step': step + 1, 'G': G.state_dict(), 'skel_enc': skel_enc.state_dict(),
                'optim': optim.state_dict(), 'rng': rng.get_state(),
                'skels': skels, 'skel_to_id': skel_to_id,
                'n_clusters': N_CLUSTERS, 'clusters': CLUSTERS,
                'n_exact_actions': N_EXACT_ACTIONS, 'exact_actions': EXACT_ACTIONS,
            }, tmp)
            os.replace(tmp, ck)
            for old in sorted(save_dir.glob('ckpt_step*.pt'))[:-2]:
                old.unlink()

    # Final ckpt
    final = save_dir / 'ckpt_final.pt'
    tmp = final.with_suffix('.pt.tmp')
    torch.save({
        'step': args.max_steps, 'G': G.state_dict(), 'skel_enc': skel_enc.state_dict(),
        'args': vars(args), 'skels': skels, 'skel_to_id': skel_to_id,
        'n_clusters': N_CLUSTERS, 'clusters': CLUSTERS,
        'n_exact_actions': N_EXACT_ACTIONS, 'exact_actions': EXACT_ACTIONS,
    }, tmp)
    os.replace(tmp, final)
    for stale in save_dir.glob('ckpt_step*.pt'):
        stale.unlink()
    print(f"\nSaved final ckpt: {final}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default='anchor_label_flow_v1')
    parser.add_argument('--max_steps', type=int, default=50000)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--warmup', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--p_cluster_drop', type=float, default=0.10,
                        help='Independent CFG dropout for cluster label (set to NULL)')
    parser.add_argument('--p_exact_drop', type=float, default=0.10,
                        help='Independent CFG dropout for exact-action label (set to NULL)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_interval', type=int, default=200)
    parser.add_argument('--ckpt_interval', type=int, default=2500)
    parser.add_argument('--no_resume', action='store_true')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
