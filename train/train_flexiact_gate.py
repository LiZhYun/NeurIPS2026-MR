"""Pilot B — FlexiAct-style timestep-dependent z-gate fine-tune on B1.

Loads a B1 behavior-conditioned checkpoint, injects the TimestepGate MLP, then
trains ONLY the gate while the backbone stays frozen. Loss is the same x_start
MSE used in train/train_behavior.py — NO cycle/counterfactual terms so this is
apples-to-apples with B1.

At each step we also log the matched-vs-null loss gap (C6 diagnostic) on the
same (x_t, t) to track whether the decoder is responding to behavior.

Usage:
    conda run -n anytop python -m train.train_flexiact_gate \
        --save_dir save/B1_flexiact_gate \
        --b1_ckpt save/B1_scratch_seed42/model000200000.pt \
        --num_steps 200 --batch_size 4 --lr 1e-3
"""
from __future__ import annotations

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import os
import json
import time
import argparse
from os.path import join as pjoin

import numpy as np
import torch

import sys
ROOT = str(PROJECT_ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.fixseed import fixseed
from utils import dist_util
from utils.model_util import create_gaussian_diffusion
from model.anytop_behavior_flexigate import AnyTopBehaviorFlexiGate
from data_loaders.get_data_conditioned import get_dataset_loader_conditioned
from diffusion.resample import UniformSampler
from train.train_behavior import ACTION_CLASSES, ACTION_TO_IDX, build_psi_lookup


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--save_dir', default='save/B1_flexiact_gate')
    p.add_argument('--b1_ckpt',
                   default='save/B1_scratch_seed42/model000200000.pt')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--num_steps', type=int, default=200)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--num_frames', type=int, default=120)
    p.add_argument('--temporal_window', type=int, default=31)
    p.add_argument('--t5_name', type=str, default='t5-base')
    p.add_argument('--effect_cache', type=str,
                   default='eval/results/effect_cache/psi_all.npy')
    p.add_argument('--clip_metadata', type=str,
                   default='eval/results/effect_cache/clip_metadata.json')
    p.add_argument('--save_interval', type=int, default=100)
    p.add_argument('--log_interval', type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    fixseed(args.seed)
    dist_util.setup_dist(args.device)
    device = dist_util.dev()
    os.makedirs(args.save_dir, exist_ok=True)
    with open(pjoin(args.save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # ---------- 1. Build flexigate model & load B1 ----------------------
    from data_loaders.truebones.truebones_utils.get_opt import get_opt
    opt = get_opt(args.device)

    b1_args_path = os.path.join(os.path.dirname(args.b1_ckpt), 'args.json')
    with open(b1_args_path) as f:
        b1_args = json.load(f)

    model = AnyTopBehaviorFlexiGate(
        max_joints=opt.max_joints, feature_len=opt.feature_len,
        latent_dim=b1_args['latent_dim'], ff_size=b1_args['latent_dim'] * 4,
        num_layers=b1_args['layers'], num_heads=4, t5_out_dim=768,
        n_actions=len(ACTION_CLASSES),
        n_behavior_tokens=b1_args.get('n_behavior_tokens', 8),
        use_residual=b1_args.get('use_residual', False),
        behavior_drop_prob=b1_args.get('behavior_drop_prob', 0.1),
        skip_t5=False, cond_mode='object_type', cond_mask_prob=0.1,
    )
    ckpt = torch.load(args.b1_ckpt, map_location='cpu', weights_only=False)
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Expect only gate.* to be missing (the rest should match B1 exactly).
    gate_missing = [m for m in missing if m.startswith('gate.')]
    other_missing = [m for m in missing if not m.startswith('gate.')]
    print(f"B1 load: missing={len(missing)} ({len(gate_missing)} gate.*, "
          f"{len(other_missing)} other)  unexpected={len(unexpected)}")
    if other_missing:
        print(f"  Non-gate missing: {other_missing[:5]}")

    # ---------- 2. Freeze everything except the gate --------------------
    for p_ in model.parameters():
        p_.requires_grad_(False)
    gate_params = list(model.gate.parameters())
    for p_ in gate_params:
        p_.requires_grad_(True)

    n_trainable = sum(p.numel() for p in gate_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Gate trainable params: {n_trainable}  "
          f"(total {n_total/1e6:.2f}M, ratio {100*n_trainable/n_total:.4f}%)")

    model.to(device)
    model.train()  # enables dropout/layernorm in the backbone behavior path
    # but gradients flow only to gate.*

    # ---------- 3. Diffusion + data -------------------------------------
    class DiffArgs:
        noise_schedule = 'cosine'
        sigma_small = True
        diffusion_steps = 100
        lambda_fs = 0.0
        lambda_geo = 0.0
    diffusion = create_gaussian_diffusion(DiffArgs())
    schedule_sampler = UniformSampler(diffusion)

    print(f"Loading ψ cache from {args.effect_cache}")
    psi_lookup, metadata = build_psi_lookup(args.clip_metadata, args.effect_cache)
    fname_to_action = {m['fname']: ACTION_TO_IDX.get(m['coarse_label'], ACTION_TO_IDX['other'])
                       for m in metadata}

    data = get_dataset_loader_conditioned(
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        temporal_window=args.temporal_window,
        t5_name=args.t5_name, balanced=False, objects_subset='all')

    optimizer = torch.optim.Adam(gate_params, lr=args.lr)

    log_path = pjoin(args.save_dir, 'train_log.jsonl')
    logf = open(log_path, 'w')

    data_iter = iter(data)
    running = dict(total=0.0, matched=0.0, null=0.0, gap=0.0, n=0)
    t0 = time.time()
    n_psi_miss = 0

    def _fill_psi(cond, bs):
        nonlocal n_psi_miss
        psi_batch, action_batch = [], []
        source_names = cond['y'].get('source_name', None)
        for bi in range(bs):
            fname = source_names[bi] if source_names is not None else None
            if fname is not None and fname in psi_lookup:
                psi_batch.append(psi_lookup[fname])
                action_batch.append(fname_to_action.get(fname, ACTION_TO_IDX['other']))
            else:
                n_psi_miss += 1
                psi_batch.append(np.zeros((64, 62), dtype=np.float32))
                action_batch.append(ACTION_TO_IDX['other'])
        cond['y']['psi'] = torch.tensor(np.stack(psi_batch),
                                        dtype=torch.float32, device=device)
        cond['y']['action_label'] = torch.tensor(action_batch,
                                                 dtype=torch.long, device=device)

    # ---------- 4. Training loop ----------------------------------------
    for step in range(args.num_steps):
        try:
            batch, cond = next(data_iter)
        except StopIteration:
            data_iter = iter(data)
            batch, cond = next(data_iter)

        batch = batch.to(device)
        for k, v in cond['y'].items():
            if torch.is_tensor(v):
                cond['y'][k] = v.to(device)
        _fill_psi(cond, batch.shape[0])

        # DDPM training step (apples-to-apples with train_behavior.py)
        t, weights = schedule_sampler.sample(batch.shape[0], device)
        losses = diffusion.training_losses(model, batch, t, model_kwargs=cond)
        loss = (losses['loss'] * weights).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gate_params, 1.0)
        optimizer.step()

        # ---- C6 diagnostic: matched vs null conditioning gap ----
        with torch.no_grad():
            # Matched: same psi/action (= just re-run forward at same (x_t, t))
            noise = torch.randn_like(batch)
            x_t = diffusion.q_sample(batch, t, noise=noise)
            lengths_mask = cond['y']['lengths_mask']
            lengths = cond['y']['lengths']
            actual_joints = cond['y']['n_joints']
            joints_mask = cond['y']['joints_mask'][:, :, :, 1, 1:]

            mo_match = model(x_t, diffusion._scale_timesteps(t), **cond)
            l_match = diffusion.temporal_spatial_masked_l2(
                batch, mo_match, lengths_mask, joints_mask, lengths, actual_joints
            ).mean().item()

            cond_null = {'y': {k: v for k, v in cond['y'].items()}}
            cond_null['y']['psi'] = torch.zeros_like(cond['y']['psi'])
            cond_null['y']['action_label'] = torch.full_like(
                cond['y']['action_label'], ACTION_TO_IDX['other'])
            mo_null = model(x_t, diffusion._scale_timesteps(t), **cond_null)
            l_null = diffusion.temporal_spatial_masked_l2(
                batch, mo_null, lengths_mask, joints_mask, lengths, actual_joints
            ).mean().item()

            gap = (l_null - l_match) / max(abs(l_null), 1e-8)

        running['total']   += float(loss.item())
        running['matched'] += l_match
        running['null']    += l_null
        running['gap']     += gap
        running['n']       += 1

        if (step + 1) % args.log_interval == 0 or step == 0:
            n = max(running['n'], 1)
            avg_total   = running['total']   / n
            avg_matched = running['matched'] / n
            avg_null    = running['null']    / n
            avg_gap     = running['gap']     / n
            elapsed = time.time() - t0
            rec = {
                'step': step + 1,
                'loss_recon': avg_total,
                'l_matched': avg_matched,
                'l_null': avg_null,
                'null_gap': avg_gap,
                'elapsed_sec': elapsed,
            }
            print(f"[{step+1}/{args.num_steps}] recon={avg_total:.4f}  "
                  f"matched={avg_matched:.4f}  null={avg_null:.4f}  "
                  f"gap={avg_gap*100:+.2f}%  psi_miss={n_psi_miss}  t={elapsed:.1f}s")
            logf.write(json.dumps(rec) + '\n'); logf.flush()
            running = dict(total=0.0, matched=0.0, null=0.0, gap=0.0, n=0)

        if (step + 1) % args.save_interval == 0 or (step + 1) == args.num_steps:
            ckpt_path = pjoin(args.save_dir, f'model_{step+1:06d}.pt')
            gate_state = {k: v.detach().cpu()
                          for k, v in model.gate.state_dict().items()}
            torch.save({'gate': gate_state, 'step': step + 1,
                        'args': vars(args)}, ckpt_path)
            print(f"  saved {ckpt_path}")

    # Dump learned g(t) curve over full schedule for easy inspection
    with torch.no_grad():
        ts = torch.arange(diffusion.num_timesteps, device=device)
        g_curve = model.gate(ts).squeeze(-1).cpu().numpy().tolist()
    with open(pjoin(args.save_dir, 'g_of_t.json'), 'w') as f:
        json.dump({'num_timesteps': diffusion.num_timesteps, 'g': g_curve}, f, indent=2)
    print(f"  g(t) curve saved to {args.save_dir}/g_of_t.json")

    logf.close()
    print("done.")


if __name__ == '__main__':
    main()
