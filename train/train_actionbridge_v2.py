"""Train VQ-ActionBridge v2: source-conditioned flow matching.

v2 differences from v1:
  - Encoder E(z_src, src_skel_id, src_graph) → behavior_tokens (replaces fixed action_emb)
  - Generator uses behavior_tokens as conditioning (cross-attention)
  - Joint training: flow MSE + action CE on encoder
  - Skel-id dropout (30%) trains for held-out generalization
  - Behavior CFG dropout (10%)

Training pairs sampled as:
  source: random row from train_v3 cache
  target: row with SAME action label (any skel) — encourages encoder skel-invariance
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
from model.actionbridge.encoder import ActionBridgeEncoder, count_parameters as count_enc
from model.actionbridge.generator_v2 import ActionBridgeGeneratorV2, count_parameters as count_g
from model.moreflow.skel_graph import (
    SkelGraphEncoder, build_skel_features, pad_to_max_joints,
)

DATA_ROOT = PROJECT_ROOT / 'dataset/truebones/zoo/truebones_processed'
COND_PATH = DATA_ROOT / 'cond.npy'
CACHE_PATH = PROJECT_ROOT / 'save/moreflow_flow/cache_train_v3.pt'
SAVE_ROOT = PROJECT_ROOT / 'save/actionbridge'

CLUSTERS = sorted(ACTION_CLUSTERS.keys())
ACTION_TO_IDX = {c: i + 1 for i, c in enumerate(CLUSTERS)}
N_CLUSTERS = len(CLUSTERS) + 1  # +1 for NULL/other (idx 0)


def lr_lambda(step, warmup, total):
    if warmup > 0 and step < warmup:
        return float(step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def build_action_index(cache, skel_to_id):
    """Build {action_idx: list of (skel_name, row_idx)} for source-target sampling."""
    by_action = defaultdict(list)
    rows = []
    skipped = 0
    for skel_name, data in cache.items():
        if skel_name.startswith('_'): continue
        meta = data['meta']
        for ri, (fname, _) in enumerate(meta):
            action = parse_action_from_filename(fname)
            cluster = action_to_cluster(action)
            if cluster is None:
                skipped += 1; continue
            aid = ACTION_TO_IDX[cluster]
            by_action[aid].append((skel_name, ri))
            rows.append((skel_name, ri, aid))
    print(f"Built {len(rows)} rows, {skipped} skipped (unmappable action)")
    print(f"Per-action counts: {[(CLUSTERS[i-1], len(by_action[i])) for i in sorted(by_action.keys()) if i > 0]}")
    return rows, by_action


def train(args):
    save_dir = SAVE_ROOT / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / 'training_log.jsonl'

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}, run_name: {args.run_name}")

    print(f"Loading cache: {CACHE_PATH}")
    cache = torch.load(CACHE_PATH, map_location='cpu', weights_only=False)
    skels = sorted([s for s in cache.keys() if not s.startswith('_')])
    skel_to_id = {s: i for i, s in enumerate(skels)}
    print(f"Cache: {len(skels)} skels")

    rows, by_action = build_action_index(cache, skel_to_id)

    # GPU pin caches
    z_per_skel = {s: cache[s]['z_continuous'].float().to(device) for s in skels}

    # Skel features
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
    skel_enc_g = SkelGraphEncoder().to(device)

    # Encoder + Generator
    E = ActionBridgeEncoder(n_skels=len(skels), n_clusters=N_CLUSTERS,
                            d_model=args.d_model_enc, n_layers=args.n_layers_enc).to(device)
    G = ActionBridgeGeneratorV2(n_skels=len(skels),
                                d_model=args.d_model_gen, n_layers=args.n_layers_gen).to(device)
    print(f"Encoder: {count_enc(E)/1e6:.1f}M, Generator: {count_g(G)/1e6:.1f}M, "
          f"SkelGraphEnc: {count_enc(skel_enc_g)/1e6:.1f}M")

    params = list(E.parameters()) + list(G.parameters()) + list(skel_enc_g.parameters())
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.LambdaLR(optim,
        lambda s: lr_lambda(s, args.warmup, args.max_steps))

    with open(save_dir / 'args.json', 'w') as f:
        json.dump({**vars(args), 'n_skels': len(skels), 'skels': skels,
                   'n_clusters': N_CLUSTERS, 'clusters': CLUSTERS,
                   'max_J': max_J}, f, indent=2)

    rng = np.random.RandomState(args.seed + 1)
    start_step = 0
    ckpts = sorted(save_dir.glob('ckpt_step*.pt'))
    if ckpts and not args.no_resume:
        latest = ckpts[-1]
        print(f"Resuming from {latest.name}")
        sd = torch.load(latest, map_location=device, weights_only=False)
        E.load_state_dict(sd['E']); G.load_state_dict(sd['G'])
        skel_enc_g.load_state_dict(sd['skel_enc_g'])
        optim.load_state_dict(sd['optim'])
        start_step = sd['step']
        rng.set_state(sd['rng'])

    print(f"Training {args.max_steps - start_step} steps from {start_step}")
    t0 = time.time()
    E.train(); G.train(); skel_enc_g.train()
    losses_window, action_acc_window = [], []

    for step in range(start_step, args.max_steps):
        optim.zero_grad()
        # Sample batch via cross-skel paired action sampling:
        # for each batch item, pick action; sample (src_skel, src_row) and (tgt_skel, tgt_row) with that action
        # 50% same skel (within-skel) + 50% diff skel (cross-skel) — encourages skel invariance
        valid_actions = [a for a in by_action if a > 0 and len(by_action[a]) >= 2]
        action_choices = rng.choice(valid_actions, args.batch_size)
        groups_by_pair = defaultdict(list)  # (src_skel, tgt_skel) → [(src_ri, tgt_ri, aid), ...]
        for aid in action_choices:
            cands = by_action[aid]
            si, ti = rng.choice(len(cands), 2, replace=False)
            src_skel, src_ri = cands[si]
            tgt_skel, tgt_ri = cands[ti]
            groups_by_pair[(src_skel, tgt_skel)].append((src_ri, tgt_ri, aid))

        total_loss_flow = 0.0; total_loss_act = 0.0
        n_correct = 0; n_total = 0
        n_micro = 0
        for (src_skel, tgt_skel), items in groups_by_pair.items():
            src_ris = torch.tensor([it[0] for it in items], dtype=torch.long, device=device)
            tgt_ris = torch.tensor([it[1] for it in items], dtype=torch.long, device=device)
            aids = torch.tensor([it[2] for it in items], dtype=torch.long, device=device)
            B = len(items)
            z_src = z_per_skel[src_skel][src_ris]   # [B, 8, 256]
            z_tgt = z_per_skel[tgt_skel][tgt_ris]   # [B, 8, 256]

            # Skel features
            sid_t = torch.full((B,), skel_to_id[src_skel], dtype=torch.long, device=device)
            tid_t = torch.full((B,), skel_to_id[tgt_skel], dtype=torch.long, device=device)
            sg = skel_enc_g(pj_padded[src_skel].unsqueeze(0), pj_mask[src_skel].unsqueeze(0),
                            pj_agg[src_skel].unsqueeze(0)).expand(B, -1)
            tg = skel_enc_g(pj_padded[tgt_skel].unsqueeze(0), pj_mask[tgt_skel].unsqueeze(0),
                            pj_agg[tgt_skel].unsqueeze(0)).expand(B, -1)

            # Skel-id dropout for held-out generalization (30%)
            drop_src = (torch.rand(1).item() < args.p_skel_drop)
            drop_tgt = (torch.rand(1).item() < args.p_skel_drop)
            # CFG drop on behavior tokens (10%)
            drop_bt = (torch.rand(1).item() < args.p_behavior_drop)

            # Encode source
            behavior_tokens, action_logits = E(z_src, sid_t, sg, drop_skel=drop_src)
            loss_action = F.cross_entropy(action_logits, aids)
            with torch.no_grad():
                n_correct += int((action_logits.argmax(-1) == aids).sum().item())
                n_total += B

            # Flow matching: target z conditioned on (skel_b, behavior_tokens)
            z0 = torch.randn_like(z_tgt)
            q = torch.rand(B, device=device)
            zt = (1 - q.view(-1, 1, 1)) * z0 + q.view(-1, 1, 1) * z_tgt
            v_target = z_tgt - z0

            v_pred = G(zt, q, tid_t, tg, behavior_tokens,
                       drop_skel=drop_tgt, drop_behavior=drop_bt)
            loss_flow = F.mse_loss(v_pred, v_target)

            loss = loss_flow + args.w_action * loss_action
            loss.backward()
            total_loss_flow += loss_flow.item()
            total_loss_act += loss_action.item()
            n_micro += 1

        if n_micro == 0: continue
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optim.step()
        sched.step()

        avg_flow = total_loss_flow / n_micro
        avg_act = total_loss_act / n_micro
        act_acc = n_correct / max(n_total, 1)
        losses_window.append(avg_flow)
        action_acc_window.append(act_acc)
        if len(losses_window) > 200:
            losses_window = losses_window[-200:]; action_acc_window = action_acc_window[-200:]

        if (step + 1) % args.log_interval == 0:
            elapsed = time.time() - t0
            eta = elapsed / max(1, step + 1 - start_step) * (args.max_steps - step - 1)
            ravg = np.mean(losses_window); racc = np.mean(action_acc_window)
            cur_lr = sched.get_last_lr()[0]
            print(f"  [{step+1:6d}/{args.max_steps}] flow={avg_flow:.3f} act={avg_act:.3f} acc={act_acc:.3f} "
                  f"avg200_flow={ravg:.3f} avg200_acc={racc:.3f} lr={cur_lr:.2e} "
                  f"({elapsed:.0f}s, ETA {eta/60:.0f}min)")
            with open(log_path, 'a') as f:
                f.write(json.dumps({'step': step+1, 'flow': avg_flow, 'act': avg_act,
                                    'act_acc': act_acc, 'avg200_flow': float(ravg),
                                    'avg200_acc': float(racc), 'lr': cur_lr}) + '\n')

        if (step + 1) % args.ckpt_interval == 0 and (step + 1) < args.max_steps:
            ck = save_dir / f'ckpt_step{step+1:07d}.pt'
            tmp = ck.with_suffix('.pt.tmp')
            torch.save({
                'step': step+1, 'E': E.state_dict(), 'G': G.state_dict(),
                'skel_enc_g': skel_enc_g.state_dict(),
                'optim': optim.state_dict(), 'rng': rng.get_state(),
                'skels': skels, 'skel_to_id': skel_to_id,
                'n_clusters': N_CLUSTERS, 'clusters': CLUSTERS,
            }, tmp)
            os.replace(tmp, ck)
            for old in sorted(save_dir.glob('ckpt_step*.pt'))[:-2]:
                old.unlink()

    final = save_dir / 'ckpt_final.pt'
    tmp = final.with_suffix('.pt.tmp')
    torch.save({
        'step': args.max_steps, 'E': E.state_dict(), 'G': G.state_dict(),
        'skel_enc_g': skel_enc_g.state_dict(),
        'args': vars(args), 'skels': skels, 'skel_to_id': skel_to_id,
        'n_clusters': N_CLUSTERS, 'clusters': CLUSTERS,
    }, tmp)
    os.replace(tmp, final)
    for stale in save_dir.glob('ckpt_step*.pt'):
        stale.unlink()
    print(f"\nSaved final: {final}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default='actionbridge_v2')
    parser.add_argument('--max_steps', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--warmup', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--d_model_enc', type=int, default=384)
    parser.add_argument('--n_layers_enc', type=int, default=4)
    parser.add_argument('--d_model_gen', type=int, default=384)
    parser.add_argument('--n_layers_gen', type=int, default=6)
    parser.add_argument('--w_action', type=float, default=0.5)
    parser.add_argument('--p_skel_drop', type=float, default=0.30)
    parser.add_argument('--p_behavior_drop', type=float, default=0.10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_interval', type=int, default=200)
    parser.add_argument('--ckpt_interval', type=int, default=2500)
    parser.add_argument('--no_resume', action='store_true')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
