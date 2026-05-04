"""Generate evaluation samples from a conditioned AnyTop model.

For each benchmark skeleton, encodes the first GT motion as source,
then generates num_reps samples on the same skeleton (self-reconstruction).
Output format matches eval_truebones.py expectations.

Usage:
    conda run -n anytop python -m eval.gen_conditioned_eval \
        --model_path save/A1v5_znorm_rank_bs_4_latentdim_256/model000175000.pt \
        --output_dir eval/results/A1v5_rank_eval \
        --benchmark eval/benchmarks/benchmark_all.txt \
        --num_reps 20
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
from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np
from data_loaders.tensors import truebones_batch_collate, create_padded_relation
from data_loaders.truebones.data.dataset import create_temporal_mask_for_window
from model.conditioners import T5Conditioner
from sample.generate_conditioned import (
    build_source_tensors, build_target_condition, encode_joints_names,
    load_args_from_checkpoint,
)
from os.path import join as pjoin


def main():
    parser = ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--benchmark', default='eval/benchmarks/benchmark_all.txt')
    parser.add_argument('--num_reps', default=20, type=int)
    parser.add_argument('--motion_length', default=6.0, type=float)
    parser.add_argument('--seed', default=10, type=int)
    parser.add_argument('--device', default=0, type=int)
    args = parser.parse_args()

    fixseed(args.seed)
    dist_util.setup_dist(args.device)

    # Load skeleton list
    with open(args.benchmark) as f:
        skeletons = [line.strip() for line in f if line.strip()]
    print(f"Benchmark: {len(skeletons)} skeletons")

    # Load model
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

    total_generated = 0
    for skel in skeletons:
        if skel not in cond_dict:
            print(f"  SKIP {skel} (not in cond_dict)")
            continue

        print(f"\n=== {skel} ===")

        # Encode first GT motion as source
        source_motion, source_offsets, source_mask = build_source_tensors(
            skel, cond_dict, opt, n_frames, ckpt.temporal_window)
        source_motion = source_motion.to(dist_util.dev())
        source_offsets = source_offsets.to(dist_util.dev())
        source_mask = source_mask.to(dist_util.dev())

        with torch.no_grad():
            enc_out = model.encoder(source_motion, source_offsets, source_mask)
            if isinstance(enc_out, tuple):
                z = enc_out[0]
            else:
                z = enc_out
        print(f"  z shape: {z.shape}, z norm: {z.norm():.3f}")

        # Build target condition (same skeleton = self-reconstruction)
        _, model_kwargs = build_target_condition(
            skel, cond_dict, n_frames, ckpt.temporal_window,
            t5_conditioner, opt.max_joints, opt.feature_len)
        model_kwargs['y']['z'] = z

        tgt = cond_dict[skel]
        n_joints = len(tgt['parents'])

        for rep_i in tqdm(range(args.num_reps), desc=f"  {skel}"):
            sample = diffusion.p_sample_loop(
                model,
                (1, opt.max_joints, opt.feature_len, n_frames),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,
                init_image=None,
                progress=False,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )

            motion = sample[0, :n_joints].cpu().permute(2, 0, 1).numpy()  # [T, J, 13]
            motion = motion * tgt['std'][None, :] + tgt['mean'][None, :]

            npy_path = pjoin(args.output_dir, f'{skel}_rep_{rep_i}_#0.npy')
            np.save(npy_path, motion)
            total_generated += 1

    print(f"\nDone. Generated {total_generated} samples in {args.output_dir}")


if __name__ == '__main__':
    main()
