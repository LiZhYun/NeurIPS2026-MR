"""Train the behavior-conditioned cross-skeleton retargeting model.

Self-supervised training: source x → R(x) = behavior_tokens (a, ψ, [r]) → AnyTopBehavior
reconstructs x on its own skeleton via DDPM.

Per refine-logs/effect_program/EMPIRICAL_PLAN.md (post Round 7 fixes):
- ONE shared model (not per-skeleton)
- B(x) = (action_label, ψ, [residual]) — all topology-normalized
- Backbone initialized from pretrained AnyTop for fast convergence
- 3 seeds, ~60 GPU-hours each on 4070 SUPER 12GB

Usage:
    conda run -n anytop python -m train.train_behavior \
        --save_dir save/B1_main_seed42 --seed 42 \
        --num_steps 200000 --batch_size 4
"""
import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from os.path import join as pjoin

from utils.fixseed import fixseed
from utils import dist_util
from utils.model_util import create_gaussian_diffusion
from model.anytop_behavior import AnyTopBehavior
from data_loaders.get_data_conditioned import get_dataset_loader_conditioned
from data_loaders.truebones.truebones_utils.get_opt import get_opt
from diffusion.resample import UniformSampler

ACTION_CLASSES = ['walk', 'run', 'idle', 'attack', 'fly', 'swim', 'jump',
                  'turn', 'die', 'eat', 'getup', 'other']
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_CLASSES)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--save_dir', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--num_steps', type=int, default=200000)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--latent_dim', type=int, default=256)
    p.add_argument('--layers', type=int, default=8)
    p.add_argument('--num_frames', type=int, default=120)
    p.add_argument('--temporal_window', type=int, default=31)
    p.add_argument('--t5_name', type=str, default='t5-base')
    p.add_argument('--init_from', type=str, default='save/all_model_dataset_truebones_bs_16_latentdim_128/model000459999.pt',
                   help='Init AnyTop backbone from checkpoint (set to "" for random)')
    p.add_argument('--use_residual', action='store_true')
    p.add_argument('--n_behavior_tokens', type=int, default=8)
    p.add_argument('--behavior_drop_prob', type=float, default=0.1)
    # Ablation flags: zero out a B component to isolate its contribution
    p.add_argument('--ablate_psi', action='store_true', help='Zero out ψ → text-only baseline')
    p.add_argument('--ablate_action', action='store_true', help='Set action label to "other" → analytic-only baseline')
    p.add_argument('--save_interval', type=int, default=10000)
    p.add_argument('--log_interval', type=int, default=100)
    p.add_argument('--effect_cache', type=str, default='eval/results/effect_cache/psi_all.npy')
    p.add_argument('--clip_metadata', type=str, default='eval/results/effect_cache/clip_metadata.json')
    p.add_argument('--smoke', action='store_true', help='Smoke-test mode: 200 steps, tiny batch')
    return p.parse_args()


def build_psi_lookup(metadata_path, psi_path):
    """Build dict: fname → psi tensor."""
    with open(metadata_path) as f:
        metadata = json.load(f)
    psi_all = np.load(psi_path)
    assert len(metadata) == len(psi_all), f"{len(metadata)} vs {len(psi_all)}"
    return {m['fname']: psi_all[i] for i, m in enumerate(metadata)}, metadata


def init_from_anytop(model, anytop_ckpt_path):
    """Copy compatible weights from pretrained AnyTop into AnyTopBehavior."""
    print(f"Loading backbone init from {anytop_ckpt_path}")
    ckpt = torch.load(anytop_ckpt_path, map_location='cpu', weights_only=False)
    model_state = model.state_dict()

    matched = 0
    skipped_shape = 0
    for k, v in ckpt.items():
        if k in model_state:
            if model_state[k].shape == v.shape:
                model_state[k] = v
                matched += 1
            else:
                skipped_shape += 1
    model.load_state_dict(model_state, strict=False)
    print(f"  Matched {matched} params, {skipped_shape} skipped (shape mismatch)")


def main():
    args = parse_args()
    if args.smoke:
        args.num_steps = 200
        args.batch_size = 2
        args.save_interval = 100
        args.log_interval = 20

    fixseed(args.seed)
    dist_util.setup_dist(args.device)
    device = dist_util.dev()
    os.makedirs(args.save_dir, exist_ok=True)

    with open(pjoin(args.save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    opt = get_opt(args.device)

    # Load ψ cache and metadata
    print(f"Loading ψ cache from {args.effect_cache}")
    psi_lookup, metadata = build_psi_lookup(args.clip_metadata, args.effect_cache)
    print(f"  {len(psi_lookup)} clips with cached ψ")

    # Build fname → action_label index lookup
    fname_to_action = {m['fname']: ACTION_TO_IDX.get(m['coarse_label'], ACTION_TO_IDX['other'])
                       for m in metadata}

    # Build model
    print(f"Building AnyTopBehavior (latent_dim={args.latent_dim}, layers={args.layers})")
    model = AnyTopBehavior(
        max_joints=opt.max_joints,
        feature_len=opt.feature_len,
        latent_dim=args.latent_dim,
        ff_size=args.latent_dim * 4,
        num_layers=args.layers,
        num_heads=4,
        t5_out_dim=768,
        n_actions=len(ACTION_CLASSES),
        n_behavior_tokens=args.n_behavior_tokens,
        use_residual=args.use_residual,
        behavior_drop_prob=args.behavior_drop_prob,
        skip_t5=False,
        cond_mode='object_type',
        cond_mask_prob=0.1,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"  Trainable params: {n_params:.2f}M")

    # Init backbone from pretrained
    if args.init_from and os.path.exists(args.init_from):
        init_from_anytop(model, args.init_from)
    else:
        print("  Random init (no backbone init)")

    model.to(device)

    # Diffusion
    class DiffArgs:
        noise_schedule = 'cosine'
        sigma_small = True
        diffusion_steps = 100
        lambda_fs = 0.0
        lambda_geo = 0.0
    diffusion = create_gaussian_diffusion(DiffArgs())
    schedule_sampler = UniformSampler(diffusion)

    # Data
    print("Building data loader...")
    data = get_dataset_loader_conditioned(
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        temporal_window=args.temporal_window,
        t5_name=args.t5_name,
        balanced=False,
        objects_subset='all',
    )

    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    print(f"\nTraining for {args.num_steps} steps, batch_size={args.batch_size}, lr={args.lr}")
    print(f"Save dir: {args.save_dir}\n")

    data_iter = iter(data)
    running_loss = 0.0
    n_logged = 0
    t0 = time.time()
    n_psi_miss = 0

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

        # Add behavior conditioning info: lookup cached ψ and action label by source_name
        bs = batch.shape[0]
        psi_batch = []
        action_batch = []
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

        psi_t = torch.tensor(np.stack(psi_batch), dtype=torch.float32, device=device)
        action_t = torch.tensor(action_batch, dtype=torch.long, device=device)
        if args.ablate_psi:
            psi_t = torch.zeros_like(psi_t)  # text-only baseline
        if args.ablate_action:
            action_t = torch.full_like(action_t, ACTION_TO_IDX['other'])  # analytic-only baseline
        cond['y']['psi'] = psi_t
        cond['y']['action_label'] = action_t

        # DDPM training step
        t, weights = schedule_sampler.sample(batch.shape[0], device)
        losses = diffusion.training_losses(model, batch, t, model_kwargs=cond)
        loss = (losses['loss'] * weights).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += loss.item()
        n_logged += 1

        if (step + 1) % args.log_interval == 0:
            elapsed = time.time() - t0
            avg = running_loss / max(n_logged, 1)
            steps_per_sec = (step + 1) / elapsed
            eta_h = (args.num_steps - step - 1) / steps_per_sec / 3600
            print(f"[{step+1}/{args.num_steps}] loss={avg:.4f}  rate={steps_per_sec:.2f}/s  "
                  f"elapsed={elapsed/60:.1f}m  eta={eta_h:.1f}h  psi_miss={n_psi_miss}")
            running_loss = 0.0
            n_logged = 0

        if (step + 1) % args.save_interval == 0:
            ckpt_path = pjoin(args.save_dir, f'model{step+1:09d}.pt')
            torch.save({'model': model.state_dict(), 'step': step + 1, 'args': vars(args)}, ckpt_path)
            print(f"  Saved {ckpt_path}")

    # Final save
    final_path = pjoin(args.save_dir, f'model{args.num_steps:09d}.pt')
    torch.save({'model': model.state_dict(), 'step': args.num_steps, 'args': vars(args)}, final_path)
    print(f"\nFinal save: {final_path}")
    print(f"Total psi extraction failures: {n_psi_miss}")


if __name__ == '__main__':
    main()
