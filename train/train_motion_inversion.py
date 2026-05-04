"""Pilot M-INV — Motion-Inversion-style fine-tuning on top of B1.

Loads the B1 checkpoint (behavior-conditioned AnyTop), injects:
  * MotionEmbed module (tiny; zero-initialized output so day-0 behavior matches B1)
  * Rank-r LoRA adapters on each STT block's temporal_attn (Q/K/V/O)

All other backbone weights are frozen. Loss = standard B1 reconstruction
(``diffusion.temporal_spatial_masked_l2`` on x-start prediction). 500 steps,
bs=4, lr=1e-4.

Usage:

    conda run -n anytop python -m train.train_motion_inversion \
        --save_dir save/B1_motion_inversion \
        --b1_ckpt save/B1_scratch_seed42/model000200000.pt \
        --num_steps 500 --batch_size 4 --lr 1e-4
"""
from __future__ import annotations

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import os
import sys
import json
import time
import argparse
from os.path import join as pjoin

import numpy as np
import torch

ROOT = str(PROJECT_ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.fixseed import fixseed
from utils import dist_util
from utils.model_util import create_gaussian_diffusion
from data_loaders.get_data_conditioned import get_dataset_loader_conditioned
from diffusion.resample import UniformSampler
from train.train_behavior import ACTION_CLASSES, ACTION_TO_IDX, build_psi_lookup
from model.anytop_motion_inversion import (
    AnyTopMotionInversion, freeze_everything_but_motion_inversion)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--save_dir', default='save/B1_motion_inversion')
    p.add_argument('--b1_ckpt', default='save/B1_scratch_seed42/model000200000.pt')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--num_steps', type=int, default=500)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--lora_rank', type=int, default=16)
    p.add_argument('--lora_alpha', type=int, default=32)
    p.add_argument('--lora_dropout', type=float, default=0.0)
    p.add_argument('--num_frames', type=int, default=120)
    p.add_argument('--temporal_window', type=int, default=31)
    p.add_argument('--t5_name', type=str, default='t5-base')
    p.add_argument('--effect_cache', type=str,
                   default='eval/results/effect_cache/psi_all.npy')
    p.add_argument('--clip_metadata', type=str,
                   default='eval/results/effect_cache/clip_metadata.json')
    p.add_argument('--save_interval', type=int, default=250)
    p.add_argument('--log_interval', type=int, default=10)
    p.add_argument('--null_gap_every', type=int, default=50)
    p.add_argument('--no_differential_v', action='store_true',
                   help='Disable differential-V transform (ablation baseline)')
    return p.parse_args()


def main():
    args = parse_args()
    fixseed(args.seed)
    dist_util.setup_dist(args.device)
    device = dist_util.dev()
    os.makedirs(args.save_dir, exist_ok=True)
    with open(pjoin(args.save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    from data_loaders.truebones.truebones_utils.get_opt import get_opt
    opt = get_opt(args.device)

    b1_args_path = os.path.join(os.path.dirname(args.b1_ckpt), 'args.json')
    with open(b1_args_path) as f:
        b1_args = json.load(f)

    model = AnyTopMotionInversion(
        max_joints=opt.max_joints, feature_len=opt.feature_len,
        latent_dim=b1_args['latent_dim'], ff_size=b1_args['latent_dim']*4,
        num_layers=b1_args['layers'], num_heads=4, t5_out_dim=768,
        n_actions=len(ACTION_CLASSES),
        n_behavior_tokens=b1_args.get('n_behavior_tokens', 8),
        use_residual=b1_args.get('use_residual', False),
        behavior_drop_prob=b1_args.get('behavior_drop_prob', 0.1),
        skip_t5=False, cond_mode='object_type', cond_mask_prob=0.1,
        n_frames=args.num_frames,
        mi_rank=args.lora_rank, mi_alpha=args.lora_alpha,
        mi_dropout=args.lora_dropout,
        use_differential_v=(not args.no_differential_v),
    )
    ckpt = torch.load(args.b1_ckpt, map_location='cpu', weights_only=False)
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    # The injected InversionTemporalMHA wraps base_mha whose params live under
    # "base" in the new module. Remap B1's temporal_attn.* → *.base.*
    remapped = {}
    for k, v in state.items():
        nk = k
        if '.temporal_attn.' in k and '.base.' not in k:
            nk = k.replace('.temporal_attn.', '.temporal_attn.base.')
        remapped[nk] = v
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    print(f"B1 load: missing={len(missing)}  unexpected={len(unexpected)}")
    if missing:
        # Sanity check: expected-new params (motion_embed, LoRA A_/B_) should
        # dominate missing. Unexpected should be empty.
        new_param_keywords = ('motion_embed.', '.A_q', '.B_q', '.A_k', '.B_k',
                              '.A_v', '.B_v', '.A_o', '.B_o')
        missing_unexpected = [m for m in missing if not any(k in m for k in new_param_keywords)]
        if missing_unexpected:
            print(f"  Unexpected missing: {missing_unexpected[:5]}...")

    trainable = freeze_everything_but_motion_inversion(model)
    model.to(device)
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Motion-Inversion trainable: {n_trainable/1e6:.3f}M / {n_total/1e6:.2f}M "
          f"({100.*n_trainable/n_total:.3f}%)")

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

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    log_path = pjoin(args.save_dir, 'train_log.jsonl')
    logf = open(log_path, 'w')

    data_iter = iter(data)
    running = dict(total=0.0, n=0)
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
        cond['y']['psi'] = torch.tensor(np.stack(psi_batch), dtype=torch.float32, device=device)
        cond['y']['action_label'] = torch.tensor(action_batch, dtype=torch.long, device=device)

    # ------ training loop -------------------------------------------------
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

        model.train()
        model.set_mi_enabled(True)
        t, _w = schedule_sampler.sample(batch.shape[0], device)
        losses = diffusion.training_losses(model, batch, t, model_kwargs=cond)
        loss = (losses['loss']).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        running['total'] += float(loss.item())
        running['n']     += 1

        # periodic conditioned-vs-null gap measurement ----------------------
        null_gap = None
        if (step + 1) % args.null_gap_every == 0 or step == 0:
            with torch.no_grad():
                model.eval()
                t_probe, _ = schedule_sampler.sample(batch.shape[0], device)
                noise = torch.randn_like(batch)
                x_t = diffusion.q_sample(batch, t_probe, noise=noise)

                mo_cond = model(x_t, diffusion._scale_timesteps(t_probe), **cond)
                lengths_mask = cond['y']['lengths_mask']
                lengths = cond['y']['lengths']
                actual_joints = cond['y']['n_joints']
                joints_mask = cond['y']['joints_mask'][:, :, :, 1, 1:]
                l_cond = diffusion.temporal_spatial_masked_l2(
                    batch, mo_cond, lengths_mask, joints_mask,
                    lengths, actual_joints).mean().item()

                # Null-conditioned: zero-ψ + "other" action. This gives the
                # real null_behavior through both routes.
                cond_null = {'y': {k: v for k, v in cond['y'].items()}}
                cond_null['y']['psi'] = torch.zeros_like(cond['y']['psi'])
                cond_null['y']['action_label'] = torch.full_like(
                    cond['y']['action_label'], ACTION_TO_IDX['other'])
                mo_null = model(x_t, diffusion._scale_timesteps(t_probe), **cond_null)
                l_null = diffusion.temporal_spatial_masked_l2(
                    batch, mo_null, lengths_mask, joints_mask,
                    lengths, actual_joints).mean().item()
            null_gap = (l_null - l_cond) / max(abs(l_null), 1e-8)

        if (step + 1) % args.log_interval == 0 or step == 0:
            n = max(running['n'], 1)
            avg_total = running['total'] / n
            elapsed = time.time() - t0
            rec = {
                'step': step + 1,
                'loss_total': avg_total,
                'null_gap': null_gap,
                'elapsed_sec': elapsed,
            }
            print(f"[{step+1}/{args.num_steps}] loss={avg_total:.4f}  "
                  f"null_gap={'' if null_gap is None else f'{null_gap*100:+.2f}%'}  "
                  f"psi_miss={n_psi_miss}  t={elapsed:.1f}s")
            logf.write(json.dumps(rec) + '\n'); logf.flush()
            running = dict(total=0.0, n=0)

        if (step + 1) % args.save_interval == 0 or (step + 1) == args.num_steps:
            ckpt_path = pjoin(args.save_dir, f'lora_{step+1:06d}.pt')
            mi_state = {}
            for i, m in enumerate(model.mi_lora_mods):
                mi_state[f'layer{i}'] = {k: v.detach().cpu()
                                         for k, v in m.state_dict().items()
                                         if k.startswith(('A_', 'B_'))}
            motion_embed_state = {k: v.detach().cpu()
                                  for k, v in model.motion_embed.state_dict().items()}
            torch.save({'lora': mi_state,
                        'motion_embed': motion_embed_state,
                        'step': step + 1,
                        'args': vars(args)}, ckpt_path)
            print(f"  saved {ckpt_path}")

    logf.close()
    print("done.")


if __name__ == '__main__':
    main()
