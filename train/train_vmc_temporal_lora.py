"""Pilot A — VMC-style temporal-attention-only LoRA on top of B1.

Loads B1 checkpoint (behavior-conditioned AnyTop decoder),
injects rank-16 LoRA deltas into every decoder layer's ``temporal_attn``
(the ONLY trainable params), and fine-tunes with two losses:

  L_total = L_simple (x_start MSE with padding mask)
          + λ_res · L_res  (MSE between frame-differences of predicted
                            and GT x_start — VMC's residual-vector loss)

All other backbone weights are frozen. Spatial attention is untouched.

Usage (default: full 200 steps, bs=4):

    conda run -n anytop python -m train.train_vmc_temporal_lora \
        --save_dir save/B1_vmc_temporal_lora \
        --b1_ckpt save/B1_scratch_seed42/model000200000.pt \
        --num_steps 200 --batch_size 4 --lr 1e-4
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
from model.anytop_behavior import AnyTopBehavior
from model.vmc_lora import inject_temporal_lora, freeze_all_but_lora
from data_loaders.get_data_conditioned import get_dataset_loader_conditioned
from diffusion.resample import UniformSampler
from train.train_behavior import ACTION_CLASSES, ACTION_TO_IDX, build_psi_lookup


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--save_dir', default='save/B1_vmc_temporal_lora')
    p.add_argument('--b1_ckpt', default='save/B1_scratch_seed42/model000200000.pt')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--num_steps', type=int, default=200)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--lora_rank', type=int, default=16)
    p.add_argument('--lora_alpha', type=int, default=32)
    p.add_argument('--lora_dropout', type=float, default=0.0)
    p.add_argument('--lambda_res', type=float, default=1.0,
                   help='Weight on residual (frame-diff) loss')
    p.add_argument('--num_frames', type=int, default=120)
    p.add_argument('--temporal_window', type=int, default=31)
    p.add_argument('--t5_name', type=str, default='t5-base')
    p.add_argument('--effect_cache', type=str,
                   default='eval/results/effect_cache/psi_all.npy')
    p.add_argument('--clip_metadata', type=str,
                   default='eval/results/effect_cache/clip_metadata.json')
    p.add_argument('--save_interval', type=int, default=100)
    p.add_argument('--log_interval', type=int, default=10)
    p.add_argument('--null_gap_every', type=int, default=50,
                   help='Every N steps, compute conditioned vs null-conditioned loss gap on the current batch.')
    return p.parse_args()


def _masked_frame_diff_mse(pred, target, lengths_mask):
    """Residual-vector loss: MSE on x_t - x_{t-1} over valid frames.

    Args
        pred, target: [B, J, F, T]
        lengths_mask: [B, ..., T] or [B, T]-like
    Returns scalar tensor.
    """
    diff_pred = pred[..., 1:] - pred[..., :-1]      # [B, J, F, T-1]
    diff_tgt  = target[..., 1:] - target[..., :-1]
    sq = (diff_pred - diff_tgt) ** 2                # [B, J, F, T-1]
    # Build per-frame validity: both frame t and t-1 valid
    m = lengths_mask
    # lengths_mask is typically shape [B, 1, 1, T_frames]. Collapse to time.
    while m.dim() > 2:
        m = m.squeeze(1) if m.shape[1] == 1 else m[:, 0]
    # Now m: [B, T]
    T = sq.shape[-1] + 1
    if m.shape[-1] != T:
        m = m[..., :T]
    valid_pair = m[..., 1:].float() * m[..., :-1].float()  # [B, T-1]
    valid_pair = valid_pair[:, None, None, :]              # broadcast over J, F
    denom = valid_pair.sum().clamp(min=1.0) * sq.shape[1] * sq.shape[2]
    return (sq * valid_pair).sum() / denom


def main():
    args = parse_args()
    fixseed(args.seed)
    dist_util.setup_dist(args.device)
    device = dist_util.dev()
    os.makedirs(args.save_dir, exist_ok=True)
    with open(pjoin(args.save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # --- 1. Build base B1 model, load weights ----------------------------
    from data_loaders.truebones.truebones_utils.get_opt import get_opt
    opt = get_opt(args.device)

    b1_args_path = os.path.join(os.path.dirname(args.b1_ckpt), 'args.json')
    with open(b1_args_path) as f:
        b1_args = json.load(f)

    model = AnyTopBehavior(
        max_joints=opt.max_joints, feature_len=opt.feature_len,
        latent_dim=b1_args['latent_dim'], ff_size=b1_args['latent_dim']*4,
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
    print(f"B1 load: missing={len(missing)}  unexpected={len(unexpected)}")

    # --- 2. Inject LoRA on temporal attention (before .to(device)) ------
    lora_mods = inject_temporal_lora(
        model, rank=args.lora_rank, alpha=args.lora_alpha,
        dropout=args.lora_dropout)
    trainable = freeze_all_but_lora(model, lora_mods)
    model.to(device)
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Injected LoRA (rank={args.lora_rank}, alpha={args.lora_alpha}) into "
          f"{len(lora_mods)} temporal_attn modules. "
          f"Trainable params: {n_trainable/1e6:.3f}M / {n_total/1e6:.2f}M total "
          f"({100.*n_trainable/n_total:.3f}%)")

    # --- 3. Diffusion + data --------------------------------------------
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
    running = dict(total=0.0, simple=0.0, res=0.0, n=0)
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
        t, _w = schedule_sampler.sample(batch.shape[0], device)

        noise = torch.randn_like(batch)
        x_t = diffusion.q_sample(batch, t, noise=noise)
        model_output = model(x_t, diffusion._scale_timesteps(t), **cond)

        # l_simple — reuse diffusion's masked l2
        lengths_mask = cond['y']['lengths_mask']
        lengths = cond['y']['lengths']
        actual_joints = cond['y']['n_joints']
        joints_mask = cond['y']['joints_mask'][:, :, :, 1, 1:]
        l_simple = diffusion.temporal_spatial_masked_l2(
            batch, model_output, lengths_mask, joints_mask, lengths, actual_joints).mean()

        # residual (frame-diff) loss on x_start prediction
        l_res = _masked_frame_diff_mse(model_output, batch, lengths_mask)

        loss = l_simple + args.lambda_res * l_res

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        running['total'] += float(loss.item())
        running['simple'] += float(l_simple.item())
        running['res']   += float(l_res.item())
        running['n']     += 1

        # periodic null-conditioned gap measurement
        null_gap = None
        if (step + 1) % args.null_gap_every == 0 or step == 0:
            with torch.no_grad():
                model.eval()
                # conditioned loss on same batch/noise
                mo_cond = model(x_t, diffusion._scale_timesteps(t), **cond)
                l_cond = diffusion.temporal_spatial_masked_l2(
                    batch, mo_cond, lengths_mask, joints_mask, lengths, actual_joints).mean().item()
                # null conditioning: zero ψ, "other" action → null tokens effectively
                cond_null = {'y': {k: v for k, v in cond['y'].items()}}
                cond_null['y']['psi'] = torch.zeros_like(cond['y']['psi'])
                cond_null['y']['action_label'] = torch.full_like(cond['y']['action_label'],
                                                                 ACTION_TO_IDX['other'])
                mo_null = model(x_t, diffusion._scale_timesteps(t), **cond_null)
                l_null = diffusion.temporal_spatial_masked_l2(
                    batch, mo_null, lengths_mask, joints_mask, lengths, actual_joints).mean().item()
            null_gap = (l_null - l_cond) / max(abs(l_null), 1e-8)  # >0 means conditioning helps

        if (step + 1) % args.log_interval == 0 or step == 0:
            n = max(running['n'], 1)
            avg_total = running['total'] / n
            avg_simple = running['simple'] / n
            avg_res = running['res'] / n
            elapsed = time.time() - t0
            rec = {
                'step': step + 1,
                'loss_total': avg_total,
                'loss_simple': avg_simple,
                'loss_residual': avg_res,
                'null_gap': null_gap,
                'elapsed_sec': elapsed,
            }
            print(f"[{step+1}/{args.num_steps}] total={avg_total:.4f}  "
                  f"simple={avg_simple:.4f}  res={avg_res:.4f}  "
                  f"null_gap={'' if null_gap is None else f'{null_gap*100:+.2f}%'}  "
                  f"psi_miss={n_psi_miss}  t={elapsed:.1f}s")
            logf.write(json.dumps(rec) + '\n'); logf.flush()
            running = dict(total=0.0, simple=0.0, res=0.0, n=0)

        if (step + 1) % args.save_interval == 0 or (step + 1) == args.num_steps:
            ckpt_path = pjoin(args.save_dir, f'lora_{step+1:06d}.pt')
            lora_state = {f'layer{i}': {k: v.detach().cpu() for k, v in m.state_dict().items()
                                         if k.startswith(('A_', 'B_'))}
                          for i, m in enumerate(lora_mods)}
            torch.save({'lora': lora_state, 'step': step + 1,
                        'args': vars(args)}, ckpt_path)
            print(f"  saved {ckpt_path}")

    logf.close()
    print("done.")


if __name__ == '__main__':
    main()
