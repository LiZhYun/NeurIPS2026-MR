"""C6 gap measurement for Motion-Inversion model.

Compares conditioned-vs-null loss on a held-out batch from the same training
loader.  Mirrors the formula used in ``train/train_vmc_temporal_lora.py``:

    gap = (loss_null - loss_matched) / |loss_null|

    gap > 0  → conditioning reduces loss (decoder uses it)
    gap ≈ 0  → decoder ignores behavior conditioning (B1 failure mode)

Reports the gap for:
  (a) Pre-training: the B1 checkpoint with MI hooks injected but LoRA/motion-embed
      at their init values (so the hook is effectively inactive — should match
      pure B1's gap).
  (b) Post-training: LoRA + motion-embed loaded from the 500-step checkpoint.
"""
from __future__ import annotations

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import os
import sys
import json
import argparse
import numpy as np
import torch

ROOT = str(PROJECT_ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--b1_ckpt', default='save/B1_scratch_seed42/model000200000.pt')
    p.add_argument('--mi_ckpt', default='save/B1_motion_inversion/lora_000500.pt')
    p.add_argument('--n_batches', type=int, default=25)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--num_frames', type=int, default=120)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--out', default='save/B1_motion_inversion/c6_gap.json')
    return p.parse_args()


@torch.no_grad()
def compute_gap(model, diffusion, data_iter, n_batches, device,
                psi_lookup, fname_to_action, ACTION_TO_IDX):
    from diffusion.resample import UniformSampler
    sampler = UniformSampler(diffusion)
    vals = {'l_matched': [], 'l_null': []}

    def fill(cond, bs, zero_out=False):
        psi_batch = []; action_batch = []
        source_names = cond['y'].get('source_name', None)
        for bi in range(bs):
            fn = source_names[bi] if source_names is not None else None
            if fn in psi_lookup:
                psi_batch.append(psi_lookup[fn])
                action_batch.append(fname_to_action.get(fn, ACTION_TO_IDX['other']))
            else:
                psi_batch.append(np.zeros((64, 62), dtype=np.float32))
                action_batch.append(ACTION_TO_IDX['other'])
        psi_t = torch.tensor(np.stack(psi_batch), dtype=torch.float32, device=device)
        action_t = torch.tensor(action_batch, dtype=torch.long, device=device)
        if zero_out:
            psi_t = torch.zeros_like(psi_t)
            action_t = torch.full_like(action_t, ACTION_TO_IDX['other'])
        cond['y']['psi'] = psi_t
        cond['y']['action_label'] = action_t

    for bi in range(n_batches):
        try:
            batch, cond = next(data_iter)
        except StopIteration:
            return vals

        batch = batch.to(device)
        for k, v in cond['y'].items():
            if torch.is_tensor(v):
                cond['y'][k] = v.to(device)

        t, _w = sampler.sample(batch.shape[0], device)
        noise = torch.randn_like(batch)
        x_t = diffusion.q_sample(batch, t, noise=noise)

        # Matched conditioning
        fill(cond, batch.shape[0], zero_out=False)
        mo = model(x_t, diffusion._scale_timesteps(t), **cond)
        lengths_mask = cond['y']['lengths_mask']
        lengths = cond['y']['lengths']
        actual_joints = cond['y']['n_joints']
        joints_mask = cond['y']['joints_mask'][:, :, :, 1, 1:]
        l_m = diffusion.temporal_spatial_masked_l2(
            batch, mo, lengths_mask, joints_mask, lengths, actual_joints
        ).mean().item()

        # Null conditioning
        fill(cond, batch.shape[0], zero_out=True)
        mo_n = model(x_t, diffusion._scale_timesteps(t), **cond)
        l_n = diffusion.temporal_spatial_masked_l2(
            batch, mo_n, lengths_mask, joints_mask, lengths, actual_joints
        ).mean().item()

        vals['l_matched'].append(l_m)
        vals['l_null'].append(l_n)

    return vals


def main():
    args = parse_args()
    from utils.fixseed import fixseed
    from utils import dist_util
    from utils.model_util import create_gaussian_diffusion
    from data_loaders.truebones.truebones_utils.get_opt import get_opt
    from data_loaders.get_data_conditioned import get_dataset_loader_conditioned
    from train.train_behavior import ACTION_CLASSES, ACTION_TO_IDX, build_psi_lookup
    from model.anytop_motion_inversion import AnyTopMotionInversion

    fixseed(args.seed)
    dist_util.setup_dist(args.device)
    device = dist_util.dev()
    opt = get_opt(args.device)

    b1_args_path = os.path.join(os.path.dirname(args.b1_ckpt), 'args.json')
    with open(b1_args_path) as f:
        b1_args = json.load(f)

    def build_model(load_mi: bool):
        model = AnyTopMotionInversion(
            max_joints=opt.max_joints, feature_len=opt.feature_len,
            latent_dim=b1_args['latent_dim'], ff_size=b1_args['latent_dim']*4,
            num_layers=b1_args['layers'], num_heads=4, t5_out_dim=768,
            n_actions=12, n_behavior_tokens=b1_args.get('n_behavior_tokens', 8),
            use_residual=False,
            behavior_drop_prob=b1_args.get('behavior_drop_prob', 0.1),
            skip_t5=False, cond_mode='object_type', cond_mask_prob=0.1,
            n_frames=args.num_frames, mi_rank=16, mi_alpha=32,
            use_differential_v=True,
        )
        ckpt = torch.load(args.b1_ckpt, map_location='cpu', weights_only=False)
        state = ckpt['model']
        remapped = {}
        for k, v in state.items():
            nk = k
            if '.temporal_attn.' in k and '.base.' not in k:
                nk = k.replace('.temporal_attn.', '.temporal_attn.base.')
            remapped[nk] = v
        model.load_state_dict(remapped, strict=False)
        if load_mi:
            mi_ckpt = torch.load(args.mi_ckpt, map_location='cpu', weights_only=False)
            for i, m in enumerate(model.mi_lora_mods):
                sd = mi_ckpt['lora'][f'layer{i}']
                for k, v in sd.items():
                    getattr(m, k).data.copy_(v)
            model.motion_embed.load_state_dict(mi_ckpt['motion_embed'], strict=True)
        model.to(device)
        model.eval()
        return model

    class DiffArgs:
        noise_schedule = 'cosine'; sigma_small = True; diffusion_steps = 100
        lambda_fs = 0.0; lambda_geo = 0.0
    diffusion = create_gaussian_diffusion(DiffArgs())

    print("Loading ψ cache")
    psi_lookup, metadata = build_psi_lookup(
        'eval/results/effect_cache/clip_metadata.json',
        'eval/results/effect_cache/psi_all.npy')
    fname_to_action = {m['fname']: ACTION_TO_IDX.get(m['coarse_label'], ACTION_TO_IDX['other'])
                       for m in metadata}

    data = get_dataset_loader_conditioned(
        batch_size=args.batch_size, num_frames=args.num_frames,
        temporal_window=31, t5_name='t5-base', balanced=False, objects_subset='all')

    # Pre-training: load B1 only, leave MI module at init (scale=0.01 so near-zero effect)
    print("=== Pre-training gap (B1 + untrained MI hook) ===")
    pre = build_model(load_mi=False)
    pre.set_mi_enabled(True)  # default, but explicit
    data_iter = iter(data)
    vals_pre = compute_gap(pre, diffusion, data_iter, args.n_batches, device,
                           psi_lookup, fname_to_action, ACTION_TO_IDX)
    del pre; torch.cuda.empty_cache()

    # Post-training
    print("=== Post-training gap (B1 + trained MI 500 steps) ===")
    post = build_model(load_mi=True)
    post.set_mi_enabled(True)
    data_iter2 = iter(data)
    vals_post = compute_gap(post, diffusion, data_iter2, args.n_batches, device,
                            psi_lookup, fname_to_action, ACTION_TO_IDX)

    def summarize(vals, tag):
        lm = float(np.mean(vals['l_matched']))
        ln = float(np.mean(vals['l_null']))
        gap = (ln - lm) / max(abs(ln), 1e-8) * 100
        print(f"  {tag}: l_matched={lm:.4f}  l_null={ln:.4f}  gap={gap:+.2f}%")
        return {'l_matched': lm, 'l_null': ln, 'gap_pct': gap,
                'raw': vals}

    r = {'pre': summarize(vals_pre, 'pre '),
         'post': summarize(vals_post, 'post')}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(r, f, indent=2)
    print(f"Saved → {args.out}")


if __name__ == '__main__':
    main()
