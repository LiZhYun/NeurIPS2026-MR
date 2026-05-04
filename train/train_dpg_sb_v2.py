"""DPG-SB-v2 training: Schrödinger Bridge in MoReFlow per-skel latent space.

INCOMPLETE IMPLEMENTATION — see Round 5 Codex audit (2026-04-24):
Codex's 4 mandatory tweaks (DPG_EVOLUTION_REVIEW_CODEX.md):
  1. [DONE] SB in latent space (per-skel VQ z [8, 256]), NOT raw motion
  2. [DONE] Single SHARED conditioned bridge
  3. [PARTIAL] "Hard target-manifold anchoring: discriminator + bone/contact/jerk penalty"
     → Latent-space discriminator IS implemented (loss_d around line 272-276).
     → Bone/contact/jerk penalty on DECODED motion is NOT implemented.
     → Cycle term is hard-zeroed (line 259: l_cycle = torch.zeros(...)).
     → Therefore tweak #3 is only half-implemented and the negative DPG-SB-v2
       result (cluster_test_test procrustes 0.400 avg) does NOT refute tweak #3.
     → Fix would require: (a) real-time Stage-A decode in the forward pass,
       (b) bone/contact/jerk loss on physical motion, (c) un-zero cycle term.
       Estimate 1-2 day re-implementation + 30-60 min retraining.
  4. [DONE] Retrieval-initialized noise (init z_b = retrieved candidate z + noise)

Training:
  - Paired (z_a, z_b) tuples where action(a) == action(b) EXACTLY, skel(a) != skel(b)
  - SB loss: flow MSE from initial (retrieved + noise) to target z_b
  - Adversarial: discriminator distinguishes real z_b from generator output (latent only)
  - Cycle: currently zeroed (see above)
  - Action-balanced sampling

Architecture: BridgeGenerator (14M) + Discriminator (1.7M) on z [8, 256]
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

from data_loaders.truebones.truebones_utils.param_utils import OBJECT_SUBSETS_DICT, DATASET_DIR
from eval.benchmark_v3.action_taxonomy import (
    ACTION_CLUSTERS, parse_action_from_filename, action_to_cluster,
)
from model.dpg_sb.dpg_sb_model import BridgeGenerator, Discriminator, count_params

DATA_ROOT = Path(DATASET_DIR)
COND_PATH = DATA_ROOT / 'cond.npy'
CACHE_PATH = PROJECT_ROOT / 'save/moreflow_flow/cache_train_v3.pt'
SAVE_ROOT = PROJECT_ROOT / 'save/dpg_sb'

CLUSTERS = sorted(ACTION_CLUSTERS.keys())


def lr_lambda(step, warmup, total):
    if warmup > 0 and step < warmup:
        return float(step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def build_index(cache, train_skels):
    """Build pair index: by_exact_action -> [(skel, row_idx)]."""
    skels = sorted(s for s in cache.keys() if not s.startswith('_') and s in train_skels)
    skel_to_id = {s: i for i, s in enumerate(skels)}
    by_exact = defaultdict(list)
    by_skel_action = defaultdict(list)  # (skel, action) -> [row_idx]
    for skel in skels:
        meta = cache[skel]['meta']
        for ri, (fname, _) in enumerate(meta):
            action = parse_action_from_filename(fname)
            cluster = action_to_cluster(action)
            if cluster is None: continue
            by_exact[action].append((skel, ri))
            by_skel_action[(skel, action)].append(ri)
    exact_actions = sorted(by_exact.keys())
    exact_to_idx = {a: i for i, a in enumerate(exact_actions)}
    return skels, skel_to_id, by_exact, by_skel_action, exact_actions, exact_to_idx


def sample_pairs(by_exact, exact_to_idx, batch_size, rng,
                 same_skel_p=0.0):
    """Sample a batch of (src_skel, src_ri, tgt_skel, tgt_ri, exact_action_idx) tuples.

    Pairs come from exact-action match across different skels (strict cross-skel).
    Returns list of tuples.
    """
    valid_actions = [a for a in by_exact.keys() if len(set(s for s, _ in by_exact[a])) >= 2]
    if not valid_actions: return []
    pairs = []
    for _ in range(batch_size):
        # Action-balanced: uniform random over valid actions
        action = valid_actions[rng.randint(len(valid_actions))]
        candidates = by_exact[action]
        # pick two from different skels
        for _try in range(20):
            i, j = rng.choice(len(candidates), 2, replace=False)
            sa, sr = candidates[i]
            sb, br = candidates[j]
            if sa != sb:
                pairs.append((sa, sr, sb, br, exact_to_idx[action]))
                break
    return pairs


def get_retrieval_init(z_per_skel, by_skel_action, src_skel, src_ri, tgt_skel,
                       action: str, exclude_ri: int = None, rng=None):
    """Pick a SAME-ACTION clip from tgt_skel as the retrieval initialization (deterministic).
    If no same-action clip on tgt_skel, fall back to a random tgt_skel clip.
    """
    candidates = by_skel_action.get((tgt_skel, action), [])
    candidates = [c for c in candidates if c != exclude_ri]
    if not candidates:
        # Pull any clip on tgt_skel (less ideal init)
        all_tgt_actions = [k for k in by_skel_action if k[0] == tgt_skel]
        if not all_tgt_actions:
            return None
        ka = all_tgt_actions[rng.randint(len(all_tgt_actions)) if rng else 0]
        candidates = by_skel_action[ka]
    if not candidates: return None
    if rng is not None:
        ri = candidates[rng.randint(len(candidates))]
    else:
        ri = candidates[0]
    return z_per_skel[tgt_skel][ri]  # [8, 256]


def train(args):
    save_dir = SAVE_ROOT / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}, run: {args.run_name}")

    # Load cache
    cache = torch.load(CACHE_PATH, map_location='cpu', weights_only=False)
    train_skels = set(OBJECT_SUBSETS_DICT['train_v3'])
    skels, skel_to_id, by_exact, by_skel_action, exact_actions, exact_to_idx = \
        build_index(cache, train_skels)
    print(f"Skels: {len(skels)}, exact actions: {len(exact_actions)}")
    print(f"Top actions: {[(a, len(by_exact[a])) for a in exact_actions[:8]]}")

    # GPU pin z per skel (with per-skel z-normalization for stable training)
    z_stats = {}  # {skel: (mean, std)}
    z_per_skel = {}
    for s in skels:
        z_raw = cache[s]['z_continuous'].float()
        # per-skel mean/std over all clips
        mu = z_raw.mean(dim=0, keepdim=True)
        sigma = z_raw.std(dim=0, keepdim=True).clamp_min(1e-3)
        z_norm = (z_raw - mu) / sigma
        z_per_skel[s] = z_norm.to(device)
        z_stats[s] = (mu.to(device), sigma.to(device))
    # Save normalization stats for inference
    torch.save({s: (mu.cpu(), sigma.cpu()) for s, (mu, sigma) in z_stats.items()},
                save_dir / 'z_stats.pt')
    print(f"Z normalized per-skel. Verify: first skel norm std = {z_per_skel[skels[0]].std():.3f} (should be ~1)")

    # Build a clip-to-action map for cycle/eval
    clip_to_action = {}  # (skel, ri) -> action_str
    for skel in skels:
        meta = cache[skel]['meta']
        for ri, (fname, _) in enumerate(meta):
            action = parse_action_from_filename(fname)
            if action_to_cluster(action) is not None:
                clip_to_action[(skel, ri)] = action

    # Models
    G = BridgeGenerator(
        codebook_dim=256, n_tokens=8,
        d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
        n_skels=len(skels), n_exact_actions=len(exact_actions),
        src_layers=args.src_layers, dropout=args.dropout,
    ).to(device)
    D = Discriminator(
        codebook_dim=256, n_tokens=8,
        d_model=args.d_model_disc, n_layers=args.n_layers_disc, n_heads=4,
        n_skels=len(skels), dropout=args.dropout,
    ).to(device)
    print(f"Generator: {count_params(G)/1e6:.1f}M, Discriminator: {count_params(D)/1e6:.1f}M")

    # Optimizers
    optG = torch.optim.AdamW(G.parameters(), lr=args.lr, weight_decay=1e-5,
                              betas=(0.9, 0.99))
    optD = torch.optim.AdamW(D.parameters(), lr=args.lr * 0.5, weight_decay=1e-5,
                              betas=(0.5, 0.99))
    schedG = torch.optim.lr_scheduler.LambdaLR(
        optG, lambda s: lr_lambda(s, args.warmup, args.max_steps))

    with open(save_dir / 'args.json', 'w') as f:
        json.dump({**vars(args),
                   'n_skels': len(skels), 'skels': skels,
                   'n_exact_actions': len(exact_actions),
                   'exact_actions': exact_actions,
                   'clusters': CLUSTERS}, f, indent=2)

    rng = np.random.RandomState(args.seed + 1)
    history = []
    losses_window = {'flow': [], 'adv_g': [], 'adv_d': [], 'cycle': []}
    t0 = time.time()
    G.train(); D.train()

    for step in range(args.max_steps):
        # Sample paired batch
        pairs = sample_pairs(by_exact, exact_to_idx, args.batch_size, rng)
        if not pairs: continue

        sa_idx = torch.tensor([skel_to_id[p[0]] for p in pairs], device=device)
        sr_idx = torch.tensor([p[1] for p in pairs], device=device)
        sb_idx = torch.tensor([skel_to_id[p[2]] for p in pairs], device=device)
        br_idx = torch.tensor([p[3] for p in pairs], device=device)
        aid = torch.tensor([p[4] for p in pairs], device=device)
        actions_str = [exact_actions[p[4]] for p in pairs]

        # Get z's
        z_a_list = [z_per_skel[p[0]][p[1]] for p in pairs]
        z_b_list = [z_per_skel[p[2]][p[3]] for p in pairs]
        z_a = torch.stack(z_a_list).to(device)  # [B, 8, 256]
        z_b = torch.stack(z_b_list).to(device)  # [B, 8, 256]

        # Retrieval-init for SB (per Codex tweak #4): for each pair, pick a
        # different SAME-ACTION clip on tgt_skel as the init (different from the GT target z_b)
        # If no same-action clip on tgt_skel exists (other than the held-out target), use noise
        z_init_list = []
        for p, action in zip(pairs, actions_str):
            init_z = get_retrieval_init(
                z_per_skel, by_skel_action,
                p[0], p[1], p[2], action, exclude_ri=p[3], rng=rng)
            if init_z is None:
                init_z = torch.randn(8, 256, device=device)
            z_init_list.append(init_z)
        z_init = torch.stack(z_init_list).to(device)  # [B, 8, 256]

        # === SB loss: flow from z_init+noise to z_b ===
        # Flow matching: t in [0, 1]
        t_diff = torch.rand(z_a.shape[0], device=device)
        noise = torch.randn_like(z_b) * args.noise_scale
        # Init point: z_init mixed with noise
        z_start = z_init + noise
        # Flow path: z_t = (1-t) * z_start + t * z_b
        t_b = t_diff.view(-1, 1, 1)
        z_t = (1 - t_b) * z_start + t_b * z_b
        v_target = z_b - z_start  # flow velocity

        src_tokens = G.encode_source(z_a)
        v_pred = G(z_t, t_diff, src_tokens, aid, sa_idx, sb_idx)
        l_flow = F.mse_loss(v_pred, v_target)

        # === Adversarial: G fools D (only after warmup) ===
        if step >= args.adv_warmup:
            t_zero = torch.zeros(z_a.shape[0], device=device)
            v_one = G(z_start, t_zero, src_tokens, aid, sa_idx, sb_idx)
            z_b_pred = z_start + v_one
            d_fake = D(z_b_pred, sb_idx)
            l_adv_g = -d_fake.mean()
        else:
            z_b_pred = z_start.detach() + (z_b - z_start).detach()  # placeholder, no grad
            l_adv_g = torch.zeros(1, device=device).squeeze()

        # === Cycle loss (skipped for v2 simplicity — defer) ===
        l_cycle = torch.zeros(1, device=device).squeeze()

        # === Total G loss ===
        loss_g = l_flow + args.w_adv * l_adv_g + args.w_cycle * l_cycle

        optG.zero_grad()
        loss_g.backward()
        torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
        optG.step()
        schedG.step()

        # === D step (only after adv warmup) ===
        if step >= args.adv_warmup:
            d_real = D(z_b, sb_idx)
            d_fake = D(z_b_pred.detach(), sb_idx)
            l_d_real = F.relu(1.0 - d_real).mean()
            l_d_fake = F.relu(1.0 + d_fake).mean()
            loss_d = l_d_real + l_d_fake

            optD.zero_grad()
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
            optD.step()
        else:
            loss_d = torch.zeros(1, device=device).squeeze()

        losses_window['flow'].append(l_flow.item())
        losses_window['adv_g'].append(l_adv_g.item())
        losses_window['adv_d'].append(loss_d.item())
        losses_window['cycle'].append(l_cycle.item())
        if len(losses_window['flow']) > 200:
            for k in losses_window:
                losses_window[k] = losses_window[k][-200:]

        if step % 50 == 0:
            elapsed = time.time() - t0
            avg = {k: sum(v)/len(v) for k, v in losses_window.items() if v}
            print(f"step {step:5d}/{args.max_steps} "
                  f"l_flow={l_flow.item():.4f} l_adv_g={l_adv_g.item():.3f} "
                  f"l_d={loss_d.item():.3f} avg_flow={avg['flow']:.4f} "
                  f"elapsed={elapsed:.0f}s")

        if step > 0 and step % args.eval_every == 0:
            ckpt_path = save_dir / f'ckpt_step{step:06d}.pt'
            torch.save({
                'step': step, 'G': G.state_dict(), 'D': D.state_dict(),
            }, ckpt_path)
            print(f"  Saved {ckpt_path.name}")

    # Final
    torch.save({'step': args.max_steps, 'G': G.state_dict(), 'D': D.state_dict()},
                save_dir / 'final.pt')
    with open(save_dir / 'training_history.json', 'w') as f:
        json.dump({'final_step': args.max_steps,
                   'last_losses': {k: v[-1] if v else None for k, v in losses_window.items()}},
                   f, indent=2)
    print(f"\nDone. Final at {save_dir / 'final.pt'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default='dpg_sb_v2')
    parser.add_argument('--max_steps', type=int, default=15000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--warmup', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--d_model', type=int, default=384)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--n_heads', type=int, default=6)
    parser.add_argument('--src_layers', type=int, default=2)
    parser.add_argument('--d_model_disc', type=int, default=256)
    parser.add_argument('--n_layers_disc', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--noise_scale', type=float, default=0.5)
    parser.add_argument('--w_adv', type=float, default=0.1)
    parser.add_argument('--w_cycle', type=float, default=0.0)
    parser.add_argument('--adv_warmup', type=int, default=2000,
                        help='Skip adversarial training for first N steps; flow-only')
    parser.add_argument('--eval_every', type=int, default=2500)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
