"""Generate evaluation samples from a conditioned AnyTop model.

Two modes:
  --mode uncond: generate with null_z (unconditional, fair comparison with A3)
  --mode diverse: condition on ALL GT motions per skeleton (fair for conditioned models)

Output format matches eval_truebones.py expectations.
"""
import os
import json
import numpy as np
import torch
from argparse import ArgumentParser
from tqdm import tqdm

from utils.fixseed import fixseed
from utils import dist_util
from utils.model_util import create_conditioned_model_and_diffusion, load_model
from data_loaders.truebones.truebones_utils.get_opt import get_opt
from data_loaders.tensors import truebones_batch_collate
from data_loaders.truebones.data.dataset import create_temporal_mask_for_window
from model.conditioners import T5Conditioner
from sample.generate_conditioned import (
    build_source_tensors, build_target_condition, encode_joints_names,
    load_args_from_checkpoint,
)
from os.path import join as pjoin


def load_all_motions(skeleton_name, cond_dict, opt, n_frames, temporal_window):
    """Load ALL stored motions for a skeleton, not just the first."""
    motion_dir = opt.motion_dir
    motion_files = sorted(
        f for f in os.listdir(motion_dir)
        if f.startswith(f'{skeleton_name}_') and f.endswith('.npy'))

    max_joints = opt.max_joints
    offsets_raw = cond_dict[skeleton_name]['offsets']
    mean = cond_dict[skeleton_name]['mean']
    std = cond_dict[skeleton_name]['std'] + 1e-6
    J_src = offsets_raw.shape[0]

    all_tensors = []
    for mf in motion_files:
        raw = np.load(pjoin(motion_dir, mf))  # [T, J_src, 13]
        T = raw.shape[0]
        norm = (raw - mean[None, :]) / std[None, :]
        norm = np.nan_to_num(norm)
        if T >= n_frames:
            norm = norm[:n_frames]
        else:
            pad = np.zeros((n_frames - T, J_src, 13))
            norm = np.concatenate([norm, pad], axis=0)

        source_motion = np.zeros((n_frames, max_joints, 13))
        source_motion[:, :J_src, :] = norm
        source_offsets = np.zeros((max_joints, 3))
        source_offsets[:J_src, :] = offsets_raw

        source_motion_t = torch.tensor(source_motion).permute(1, 2, 0).float().unsqueeze(0)
        source_offsets_t = torch.tensor(source_offsets).float().unsqueeze(0)
        source_mask = torch.zeros(1, max_joints, dtype=torch.bool)
        source_mask[0, :J_src] = True

        all_tensors.append((source_motion_t, source_offsets_t, source_mask))

    return all_tensors, motion_files


def main():
    parser = ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--benchmark', default='eval/benchmarks/benchmark_all.txt')
    parser.add_argument('--mode', default='uncond', choices=['uncond', 'diverse'])
    parser.add_argument('--num_reps', default=20, type=int,
                        help='Reps per skeleton (uncond) or per source motion (diverse)')
    parser.add_argument('--motion_length', default=6.0, type=float)
    parser.add_argument('--seed', default=10, type=int)
    parser.add_argument('--device', default=0, type=int)
    args = parser.parse_args()

    fixseed(args.seed)
    dist_util.setup_dist(args.device)

    with open(args.benchmark) as f:
        skeletons = [line.strip() for line in f if line.strip()]
    print(f"Benchmark: {len(skeletons)} skeletons, mode={args.mode}")

    ckpt_args = load_args_from_checkpoint(args.model_path)

    class Namespace:
        def __init__(self, d):
            self.__dict__.update(d)
    ckpt = Namespace(ckpt_args)

    opt = get_opt(args.device)
    n_frames = int(args.motion_length * opt.fps)
    cond_dict = np.load(opt.cond_file, allow_pickle=True).item()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Creating model and diffusion...")
    model, diffusion = create_conditioned_model_and_diffusion(ckpt)
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model(model, state_dict)
    model.to(dist_util.dev())
    model.eval()

    print("Loading T5...")
    t5_conditioner = T5Conditioner(
        name=ckpt.t5_name, finetune=False, word_dropout=0.0,
        normalize_text=False, device='cuda')

    total = 0
    for skel in skeletons:
        if skel not in cond_dict:
            print(f"  SKIP {skel}")
            continue

        print(f"\n=== {skel} ===")
        _, model_kwargs = build_target_condition(
            skel, cond_dict, n_frames, ckpt.temporal_window,
            t5_conditioner, opt.max_joints, opt.feature_len)
        tgt = cond_dict[skel]
        n_joints = len(tgt['parents'])

        if args.mode == 'uncond':
            # Use null_z — unconditional generation
            null_z = model.null_z.expand(1, -1, model.encoder.num_queries, model.latent_dim)
            model_kwargs['y']['z'] = null_z

            for rep_i in tqdm(range(args.num_reps), desc=f"  {skel}"):
                sample = diffusion.p_sample_loop(
                    model, (1, opt.max_joints, opt.feature_len, n_frames),
                    clip_denoised=False, model_kwargs=model_kwargs,
                    skip_timesteps=0, progress=False)

                motion = sample[0, :n_joints].cpu().permute(2, 0, 1).numpy()
                motion = motion * tgt['std'][None, :] + tgt['mean'][None, :]
                npy_path = pjoin(args.output_dir, f'{skel}_rep_{rep_i}_#0.npy')
                np.save(npy_path, motion)
                total += 1

        elif args.mode == 'diverse':
            # Encode ALL GT motions, generate 1 sample per source
            all_motions, motion_files = load_all_motions(
                skel, cond_dict, opt, n_frames, ckpt.temporal_window)
            n_sources = min(len(all_motions), args.num_reps)

            for src_i in tqdm(range(n_sources), desc=f"  {skel}"):
                src_mot, src_off, src_mask = all_motions[src_i]
                src_mot = src_mot.to(dist_util.dev())
                src_off = src_off.to(dist_util.dev())
                src_mask = src_mask.to(dist_util.dev())

                with torch.no_grad():
                    enc_out = model.encoder(src_mot, src_off, src_mask)
                    z = enc_out[0] if isinstance(enc_out, tuple) else enc_out

                model_kwargs['y']['z'] = z

                sample = diffusion.p_sample_loop(
                    model, (1, opt.max_joints, opt.feature_len, n_frames),
                    clip_denoised=False, model_kwargs=model_kwargs,
                    skip_timesteps=0, progress=False)

                motion = sample[0, :n_joints].cpu().permute(2, 0, 1).numpy()
                motion = motion * tgt['std'][None, :] + tgt['mean'][None, :]
                npy_path = pjoin(args.output_dir, f'{skel}_rep_{src_i}_#0.npy')
                np.save(npy_path, motion)
                total += 1

    print(f"\nDone. Generated {total} samples in {args.output_dir}")


if __name__ == '__main__':
    main()
