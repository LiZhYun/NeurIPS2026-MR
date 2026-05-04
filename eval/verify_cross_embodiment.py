"""Verify cross-embodiment: same noise → same semantics across skeletons?

If AnyTop has cross-skeleton understanding, the SAME initial noise should
produce semantically similar motions on different skeletons. If each skeleton
is just an independent distribution, the motions will be unrelated.

Generates 3 noise seeds × 5 skeletons = 15 motions, then:
  1. Renders side-by-side comparison videos (all 5 skeletons per seed)
  2. Computes pairwise cosine similarity of velocity profiles across skeletons
     for the same seed (high sim = cross-skel alignment) vs different seeds (low sim = chance)

Usage:
    conda run -n anytop python -m eval.verify_cross_embodiment
"""
import os
import json
import argparse
import numpy as np
import torch
from os.path import join as pjoin


SKELETONS = ['Horse', 'Jaguar', 'Alligator', 'Anaconda', 'Parrot']
N_SEEDS = 3


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path',
                   default='save/all_model_dataset_truebones_bs_16_latentdim_128/model000459999.pt')
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--out_dir', type=str, default='eval/results/cross_embodiment')
    return p.parse_args()


def main():
    args = parse_args()
    from utils.fixseed import fixseed
    from utils import dist_util
    from utils.model_util import create_model_and_diffusion_general_skeleton, load_model
    from data_loaders.truebones.truebones_utils.get_opt import get_opt
    from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np
    from data_loaders.truebones.data.dataset import create_temporal_mask_for_window
    from data_loaders.tensors import truebones_batch_collate
    from model.conditioners import T5Conditioner

    dist_util.setup_dist(args.device)
    device = dist_util.dev()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load model
    with open(pjoin(os.path.dirname(args.model_path), 'args.json')) as f:
        args_d = json.load(f)
    class NS:
        def __init__(self, d): self.__dict__.update(d)
    m_args = NS(args_d)
    model, diffusion = create_model_and_diffusion_general_skeleton(m_args)
    state = torch.load(args.model_path, map_location='cpu')
    load_model(model, state)
    model.to(device)
    model.eval()

    opt = get_opt(args.device)
    cond_dict = np.load(opt.cond_file, allow_pickle=True).item()
    n_frames = 80  # 4 seconds at 20fps
    max_joints = opt.max_joints

    t5 = T5Conditioner(name=m_args.t5_name, finetune=False, word_dropout=0.0,
                       normalize_text=False, device='cuda')

    available = [s for s in SKELETONS if s in cond_dict]
    print(f"Skeletons: {available}")

    def encode_joints_names(names, t5_cond):
        return t5_cond(t5_cond.tokenize(names))

    def build_condition_single(skel_name):
        """Build model_kwargs for a single skeleton."""
        obj = cond_dict[skel_name]
        parents = obj['parents']
        n_joints = len(parents)
        mean = obj['mean']
        std = obj['std']
        tpos = (obj['tpos_first_frame'] - mean) / (std + 1e-6)
        tpos = np.nan_to_num(tpos)
        names_emb = encode_joints_names(obj['joints_names'], t5).detach().cpu().numpy()

        batch = [
            np.zeros((n_frames, n_joints, opt.feature_len)),
            n_frames,
            parents,
            tpos,
            obj['offsets'],
            create_temporal_mask_for_window(31, n_frames),
            obj['joints_graph_dist'],
            obj['joint_relations'],
            skel_name,
            names_emb,
            0,
            mean,
            std,
            max_joints,
        ]
        return batch

    # Generate with same noise for each seed × skeleton
    all_motions = {}  # {(seed, skel): motion_denorm}
    all_positions = {}  # {(seed, skel): global_positions}

    for seed_idx in range(N_SEEDS):
        seed = 1000 + seed_idx * 100
        print(f"\n=== Seed {seed} ===")

        # Generate fixed noise ONCE per seed
        fixseed(seed)
        fixed_noise = torch.randn(1, max_joints, opt.feature_len, n_frames, device=device)

        for skel_name in available:
            info = cond_dict[skel_name]
            n_joints = len(info['joints_names'])

            batch = build_condition_single(skel_name)
            _, model_kwargs = truebones_batch_collate([batch])
            for k, v in model_kwargs['y'].items():
                if torch.is_tensor(v):
                    model_kwargs['y'][k] = v.to(device)

            with torch.no_grad():
                sample = diffusion.p_sample_loop(
                    model,
                    (1, max_joints, opt.feature_len, n_frames),
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    noise=fixed_noise.clone(),
                    progress=False,
                )

            motion = sample[0, :n_joints].cpu().permute(2, 0, 1).numpy()  # [T, J, 13]
            mean = info['mean'][:n_joints]
            std = info['std'][:n_joints]
            motion_denorm = motion * (std + 1e-6) + mean
            positions = recover_from_bvh_ric_np(motion_denorm)  # [T, J, 3]

            all_motions[(seed_idx, skel_name)] = motion_denorm
            all_positions[(seed_idx, skel_name)] = positions

            vel = np.sqrt((np.diff(positions, axis=0)**2).sum(axis=2)).mean()
            print(f"  {skel_name:12s} ({n_joints:2d}j): vel_mag={vel:.4f}")

    # Render enhanced videos
    from eval.visualize_motion import render_motion_mp4
    for seed_idx in range(N_SEEDS):
        for skel_name in available:
            positions = all_positions[(seed_idx, skel_name)]
            parents = cond_dict[skel_name]['parents']
            n_joints = len(cond_dict[skel_name]['joints_names'])
            save_path = pjoin(args.out_dir, f'seed{seed_idx}_{skel_name}.mp4')
            render_motion_mp4(positions, parents, n_joints, save_path,
                              title=f'Seed {seed_idx} — {skel_name} ({n_joints}j)', fps=20)

    # Cross-skeleton velocity profile similarity analysis
    print("\n" + "=" * 60)
    print("CROSS-EMBODIMENT ANALYSIS: velocity profile cosine similarity")
    print("=" * 60)
    print("Same seed, different skeletons → high sim = cross-skel alignment")
    print("Different seed, same skeleton → low sim = seed-dependent variation\n")

    def velocity_profile(positions):
        """Compute per-frame mean velocity magnitude as a 1D profile."""
        vel = np.sqrt((np.diff(positions, axis=0)**2).sum(axis=2))  # [T-1, J]
        return vel.mean(axis=1)  # [T-1]

    # Same-seed cross-skeleton similarity
    same_seed_sims = []
    for seed_idx in range(N_SEEDS):
        profiles = {}
        for skel_name in available:
            vp = velocity_profile(all_positions[(seed_idx, skel_name)])
            profiles[skel_name] = vp / (np.linalg.norm(vp) + 1e-8)
        for i, s1 in enumerate(available):
            for j, s2 in enumerate(available):
                if i < j:
                    sim = float(np.dot(profiles[s1], profiles[s2]))
                    same_seed_sims.append(sim)
                    print(f"  seed={seed_idx} {s1:12s} vs {s2:12s}: cos={sim:.4f}")

    # Different-seed same-skeleton similarity
    diff_seed_sims = []
    for skel_name in available:
        profiles = {}
        for seed_idx in range(N_SEEDS):
            vp = velocity_profile(all_positions[(seed_idx, skel_name)])
            profiles[seed_idx] = vp / (np.linalg.norm(vp) + 1e-8)
        for i in range(N_SEEDS):
            for j in range(i + 1, N_SEEDS):
                sim = float(np.dot(profiles[i], profiles[j]))
                diff_seed_sims.append(sim)

    print(f"\nSame seed, diff skeleton (cross-embodiment): "
          f"mean={np.mean(same_seed_sims):.4f} ± {np.std(same_seed_sims):.4f}")
    print(f"Diff seed, same skeleton (variation):         "
          f"mean={np.mean(diff_seed_sims):.4f} ± {np.std(diff_seed_sims):.4f}")

    cross_embod = np.mean(same_seed_sims) > np.mean(diff_seed_sims) + np.std(diff_seed_sims)
    print(f"\nCross-embodiment signal: "
          f"{'YES — same noise → similar temporal dynamics across skeletons'if cross_embod else 'NO — noise does not transfer semantics across skeletons'}")

    results = {
        'skeletons': available,
        'n_seeds': N_SEEDS,
        'same_seed_cross_skel_sims': same_seed_sims,
        'diff_seed_same_skel_sims': diff_seed_sims,
        'mean_cross_embodiment': float(np.mean(same_seed_sims)),
        'mean_variation': float(np.mean(diff_seed_sims)),
        'cross_embodiment_detected': bool(cross_embod),
    }
    out_path = pjoin(args.out_dir, 'cross_embodiment_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out_path}")
    print(f"Videos → {args.out_dir}/seed*_*.mp4")


if __name__ == '__main__':
    main()
