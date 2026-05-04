"""Train ANCHOR-Label-Flow-Src: source + cluster + exact-action conditional flow.

Codex R4 final blocker: build a generator with the FULL matched input tuple
that ANCHOR has access to: source motion + predicted cluster + exact action +
target skeleton graph. If this still fails on V5 cluster-tier, the metadata-
availability defense is fully refuted and the structural identifiability claim
is airtight.

Forked from train_anchor_label_flow.py with:
- Pair sampling: (z_a, z_b) where action(a)==action(b) (exact match), skel(a)!=skel(b)
- Source motion latent z_a fed into AnchorLabelFlowSrcGenerator alongside target
- Source skeleton id + source graph features as global conditioning
- THREE independent CFG dropouts: cluster, exact, source (all default 10%)
- Loss: standard flow MSE on target velocity v = z_b - z_noise

Usage:
  python -m train.train_anchor_label_flow_src --run_name anchor_label_flow_src_v1 --max_steps 50000
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
from model.actionbridge.generator_anchor_src import (
    AnchorLabelFlowSrcGenerator, count_parameters,
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
N_CLUSTERS = len(CLUSTERS) + 1

# Exact-action vocab: index 0 = NULL (CFG drop), 1..N = real exact actions
EXACT_ACTIONS = sorted({a for acts in ACTION_CLUSTERS.values() for a in acts})
EXACT_ACTION_TO_IDX = {a: i + 1 for i, a in enumerate(EXACT_ACTIONS)}
N_EXACT_ACTIONS = len(EXACT_ACTIONS) + 1


def lr_lambda(step, warmup, total):
    if warmup > 0 and step < warmup:
        return float(step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def build_pair_index(cache, train_skels):
    """Build pair index by exact action: by_exact[action] -> [(skel, row_idx, cluster_idx, exact_idx)]."""
    skels = sorted(s for s in cache.keys() if not s.startswith('_') and s in train_skels)
    skel_to_id = {s: i for i, s in enumerate(skels)}
    by_exact = defaultdict(list)
    skipped_cluster = skipped_exact = 0
    for skel in skels:
        meta = cache[skel]['meta']
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
            by_exact[action].append((skel, ri, cid, eid))
    print(f"Pair index: {len(skels)} skels, {len(by_exact)} exact actions; "
          f"skipped {skipped_cluster} unmappable cluster, {skipped_exact} unmappable exact")
    return skels, skel_to_id, by_exact


def sample_pairs(by_exact, batch_size, rng):
    """Sample (skel_a, ri_a, skel_b, ri_b, cid, eid) tuples — paired by exact action,
    cross-skel only. Action-balanced uniform over actions with >=2 distinct skels.
    """
    valid_actions = [a for a in by_exact.keys()
                     if len(set(s for s, _, _, _ in by_exact[a])) >= 2]
    if not valid_actions:
        return []
    pairs = []
    for _ in range(batch_size):
        action = valid_actions[rng.randint(len(valid_actions))]
        candidates = by_exact[action]
        # pick two from different skels
        for _try in range(20):
            i, j = rng.choice(len(candidates), 2, replace=False)
            sa, ra, ca, ea = candidates[i]
            sb, rb, cb, eb = candidates[j]
            if sa != sb:
                # cluster + exact are identical for pair (same exact action)
                pairs.append((sa, ra, sb, rb, ca, ea))
                break
    return pairs


def train(args):
    save_dir = SAVE_ROOT / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / 'training_log.jsonl'

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}, run_name: {args.run_name}")

    # Load cache
    print(f"Loading cache: {CACHE_PATH}")
    cache = torch.load(CACHE_PATH, map_location='cpu', weights_only=False)
    train_skels = set(OBJECT_SUBSETS_DICT['train_v3'])
    skels, skel_to_id, by_exact = build_pair_index(cache, train_skels)
    if not by_exact:
        raise RuntimeError("No paired training rows after filtering")

    # Pin z per skel to GPU (raw, no per-skel z-normalization — matches AL-Flow trainer)
    z_per_skel = {s: cache[s]['z_continuous'].float().to(device) for s in skels}

    # Skel graph features
    print("Building skel-graph features...")
    cond_dict = np.load(COND_PATH, allow_pickle=True).item()
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
    skel_enc = SkelGraphEncoder().to(device)

    # Generator
    G = AnchorLabelFlowSrcGenerator(
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

    # Save args
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
        pairs = sample_pairs(by_exact, args.batch_size, rng)
        if not pairs:
            continue
        # Group by (skel_a, skel_b) to amortize skel_enc cost — but for simplicity, run per-pair-skel-batch
        # We'll group by tgt_skel since that's what determines z_b dim and skel_enc target call
        groups_by_tgt = defaultdict(list)
        for (sa, ra, sb, rb, ca, ea) in pairs:
            groups_by_tgt[sb].append((sa, ra, rb, ca, ea))
        groups_by_src = defaultdict(list)
        for sb, items in groups_by_tgt.items():
            for (sa, ra, rb, ca, ea) in items:
                groups_by_src[(sa, sb)].append((ra, rb, ca, ea))

        total_loss = 0.0
        n_micro = 0
        for (sa, sb), items in groups_by_src.items():
            ra_t = torch.tensor([it[0] for it in items], dtype=torch.long, device=device)
            rb_t = torch.tensor([it[1] for it in items], dtype=torch.long, device=device)
            ca_t = torch.tensor([it[2] for it in items], dtype=torch.long, device=device)
            ea_t = torch.tensor([it[3] for it in items], dtype=torch.long, device=device)
            B = len(items)

            z_a = z_per_skel[sa][ra_t]                      # [B, 8, 256] source latent
            z_b = z_per_skel[sb][rb_t]                      # [B, 8, 256] target latent
            z_noise = torch.randn_like(z_b)
            q = torch.rand(B, device=device)
            z_b_t = (1 - q.view(-1, 1, 1)) * z_noise + q.view(-1, 1, 1) * z_b
            v_target = z_b - z_noise

            # CFG dropouts (3 independent channels)
            cluster_drop = (torch.rand(B, device=device) < args.p_cluster_drop)
            exact_drop = (torch.rand(B, device=device) < args.p_exact_drop)
            src_drop = (torch.rand(B, device=device) < args.p_src_drop)

            tid = torch.full((B,), skel_to_id[sb], dtype=torch.long, device=device)
            sid = torch.full((B,), skel_to_id[sa], dtype=torch.long, device=device)
            tg = skel_enc(pj_padded[sb].unsqueeze(0), pj_mask[sb].unsqueeze(0),
                          pj_agg[sb].unsqueeze(0)).expand(B, -1)
            sg = skel_enc(pj_padded[sa].unsqueeze(0), pj_mask[sa].unsqueeze(0),
                          pj_agg[sa].unsqueeze(0)).expand(B, -1)

            v_pred = G(z_b_t, q, tid, tg, z_a, sid, sg, ca_t, ea_t,
                       cluster_mask=cluster_drop, exact_mask=exact_drop, src_mask=src_drop)
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
        if len(losses_window) > 200:
            losses_window = losses_window[-200:]

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
    parser.add_argument('--run_name', type=str, default='anchor_label_flow_src_v1')
    parser.add_argument('--max_steps', type=int, default=50000)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--warmup', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Smaller than AL-Flow because each item is 2x tokens (src+tgt)')
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--p_cluster_drop', type=float, default=0.10)
    parser.add_argument('--p_exact_drop', type=float, default=0.10)
    parser.add_argument('--p_src_drop', type=float, default=0.10,
                        help='CFG dropout for source motion (replace src_z with NULL token)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_interval', type=int, default=200)
    parser.add_argument('--ckpt_interval', type=int, default=2500)
    parser.add_argument('--no_resume', action='store_true')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
