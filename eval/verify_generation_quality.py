"""Verify 2: Does the pretrained AnyTop generate plausible cross-skeleton motion?

Generate unconditional samples from the pretrained all_model for representative
skeletons and compute basic sanity metrics:
  - joint velocity magnitude (non-zero = motion is not degenerate)
  - range of motion per joint (spread = skeleton-appropriate poses)
  - inter-sample diversity (different samples ≠ collapsed)

Saves .npy files for visual inspection.

Usage:
    conda run -n anytop python -m eval.verify_generation_quality
"""
import os
import json
import argparse
import numpy as np
import torch
from os.path import join as pjoin


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path',
                   default='save/all_model_dataset_truebones_bs_16_latentdim_128/model000459999.pt')
    p.add_argument('--n_samples', type=int, default=5, help='Samples per skeleton')
    p.add_argument('--seed', type=int, default=10)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--out_dir', type=str, default='eval/results/generation_quality')
    return p.parse_args()


REPRESENTATIVE_SKELETONS = [
    'Horse',       # quadruped
    'Parrot',      # flying/bird
    'Snake',       # serpentine
    'Millipede',   # many-legged
    'Jaguar',      # quadruped (different from Horse)
]


def load_pretrained_anytop(model_path, device):
    from utils.model_util import create_model_and_diffusion_general_skeleton, load_model

    with open(pjoin(os.path.dirname(model_path), 'args.json')) as f:
        args_d = json.load(f)
    class NS:
        def __init__(self, d): self.__dict__.update(d)
    args = NS(args_d)
    model, diffusion = create_model_and_diffusion_general_skeleton(args)
    state = torch.load(model_path, map_location='cpu')
    load_model(model, state)
    model.to(device)
    model.eval()
    return model, diffusion, args


def build_y_dict(cond_dict, skel_name, opt, t5, n_frames, device):
    """Build the y conditioning dict for unconditional sampling on a given skeleton."""
    from data_loaders.truebones.data.dataset import create_temporal_mask_for_window
    from data_loaders.tensors import create_padded_relation

    info = cond_dict[skel_name]
    max_joints = opt.max_joints
    feature_len = opt.feature_len
    n_joints = len(info['joints_names'])
    mean = info['mean']
    std = info['std'] + 1e-6

    tpos_raw = info['tpos_first_frame']
    tpos = np.zeros((max_joints, feature_len))
    tpos[:n_joints] = (tpos_raw - mean) / std
    tpos = np.nan_to_num(tpos)
    tpos_t = torch.tensor(tpos).float().unsqueeze(0).to(device)

    names = info['joints_names']
    names_emb = t5(t5.tokenize(names)).detach().cpu().numpy()
    names_padded = np.zeros((max_joints, names_emb.shape[1]))
    names_padded[:n_joints] = names_emb
    names_t = torch.tensor(names_padded).float().unsqueeze(0).to(device)

    gd = create_padded_relation(info['joints_graph_dist'], max_joints, n_joints)
    jr = create_padded_relation(info['joint_relations'], max_joints, n_joints)
    gd_t = torch.tensor(gd).long().unsqueeze(0).to(device)
    jr_t = torch.tensor(jr).long().unsqueeze(0).to(device)

    jmask_5d = torch.zeros(1, 1, 1, max_joints + 1, max_joints + 1, device=device)
    jmask_5d[0, 0, 0, :n_joints + 1, :n_joints + 1] = 1.0

    tmask = create_temporal_mask_for_window(31, n_frames)
    tmask_t = torch.tensor(tmask).unsqueeze(0).unsqueeze(2).unsqueeze(3).float().to(device)

    y = {
        'joints_mask':       jmask_5d,
        'mask':              tmask_t,
        'tpos_first_frame':  tpos_t,
        'joints_names_embs': names_t,
        'graph_dist':        gd_t,
        'joints_relations':  jr_t,
        'crop_start_ind':    torch.zeros(1, dtype=torch.long, device=device),
        'n_joints':          torch.tensor([n_joints]),
    }
    return y, n_joints, mean, std


@torch.no_grad()
def sample_ddpm(model, diffusion, y, shape, device):
    """Simple DDPM sampling loop."""
    x = torch.randn(shape, device=device)
    for t_val in reversed(range(diffusion.num_timesteps)):
        t = torch.tensor([t_val], device=device)
        out = diffusion.p_mean_variance(model, x, t, model_kwargs={'y': y})
        x = out['mean']
        if t_val > 0:
            noise = torch.randn_like(x)
            x = x + torch.exp(0.5 * out['log_variance']) * noise
    return x


def compute_motion_stats(motion_norm, n_joints, mean, std):
    """Compute basic sanity metrics on a generated motion [J, 13, T]."""
    m = motion_norm[:n_joints, :, :]  # [J_real, 13, T]

    pos = m[:, :3, :]  # root-relative position [J, 3, T]
    vel = m[:, 10:13, :] if m.shape[1] >= 13 else None  # velocity [J, 3, T]

    pos_range = (pos.max(dim=-1).values - pos.min(dim=-1).values).mean().item()

    if vel is not None:
        vel_mag = np.sqrt((vel ** 2).sum(axis=1)).mean()  # mean velocity magnitude
    else:
        vel_mag = 0.0

    rot6d = m[:, 3:9, :]  # 6D rotation [J, 6, T]
    rot_range = (rot6d.max(dim=-1).values - rot6d.min(dim=-1).values).mean().item()

    return {
        'pos_range': float(pos_range),
        'vel_magnitude': float(vel_mag),
        'rot_range': float(rot_range),
    }


def main():
    args = parse_args()
    from utils.fixseed import fixseed
    from utils import dist_util
    fixseed(args.seed)
    dist_util.setup_dist(args.device)
    device = dist_util.dev()

    os.makedirs(args.out_dir, exist_ok=True)

    model, diffusion, m_args = load_pretrained_anytop(args.model_path, device)
    n_frames = getattr(m_args, 'num_frames', 120)
    print(f"Loaded model. n_frames={n_frames}, latent_dim={m_args.latent_dim}")

    from data_loaders.truebones.truebones_utils.get_opt import get_opt
    opt = get_opt(args.device)
    cond_dict = np.load(opt.cond_file, allow_pickle=True).item()

    from model.conditioners import T5Conditioner
    t5 = T5Conditioner(name=m_args.t5_name, finetune=False, word_dropout=0.0,
                       normalize_text=False, device='cuda')

    available_skels = list(cond_dict.keys())
    skels_to_test = [s for s in REPRESENTATIVE_SKELETONS if s in available_skels]
    print(f"Testing skeletons: {skels_to_test}")
    if len(skels_to_test) < len(REPRESENTATIVE_SKELETONS):
        missing = set(REPRESENTATIVE_SKELETONS) - set(skels_to_test)
        print(f"  Missing (not in cond_dict): {missing}")
        for s in available_skels:
            if len(skels_to_test) >= 5:
                break
            if s not in skels_to_test:
                skels_to_test.append(s)
        print(f"  Padded to: {skels_to_test}")

    # Also load real motions for comparison
    with open('dataset/truebones/zoo/truebones_processed/motion_labels.json') as f:
        label_map = json.load(f)

    all_results = {}

    for skel_name in skels_to_test:
        print(f"\n{'='*50}")
        print(f"Skeleton: {skel_name} ({len(cond_dict[skel_name]['joints_names'])} joints)")
        print(f"{'='*50}")

        n_joints = len(cond_dict[skel_name]['joints_names'])
        y, nj, mean, std = build_y_dict(cond_dict, skel_name, opt, t5, n_frames, device)
        shape = (1, opt.max_joints, opt.feature_len, n_frames)

        # Load real motions for this skeleton
        real_files = [f for f, info in label_map.items()
                      if info['skeleton'] == skel_name][:5]
        real_stats = []
        for rf in real_files:
            raw = np.load(pjoin(opt.motion_dir, rf))
            T_raw = raw.shape[0]
            if T_raw < n_frames:
                raw = np.concatenate([raw, np.zeros((n_frames - T_raw, raw.shape[1], 13))], axis=0)
            else:
                raw = raw[:n_frames]
            norm = np.nan_to_num((raw - mean[None, :]) / (std[None, :] + 1e-6))
            norm_t = torch.tensor(norm).permute(1, 2, 0).float()  # [J, 13, T]
            st = compute_motion_stats(norm_t, nj, mean, std)
            real_stats.append(st)

        real_avg = {k: np.mean([s[k] for s in real_stats]) for k in real_stats[0]} if real_stats else {}
        print(f"  Real motions ({len(real_files)} clips): {real_avg}")

        # Generate samples
        gen_stats = []
        gen_motions = []
        for i in range(args.n_samples):
            fixseed(args.seed + i * 100)
            sample = sample_ddpm(model, diffusion, y, shape, device)
            sample_np = sample[0].cpu()  # [J, 13, T]
            gen_motions.append(sample_np.numpy())
            st = compute_motion_stats(sample_np, nj, mean, std)
            gen_stats.append(st)
            print(f"  Sample {i}: pos_range={st['pos_range']:.3f} "
                  f"vel_mag={st['vel_magnitude']:.3f} rot_range={st['rot_range']:.3f}")

        gen_avg = {k: np.mean([s[k] for s in gen_stats]) for k in gen_stats[0]}

        # Inter-sample diversity
        if len(gen_motions) >= 2:
            flat = [m[:nj].reshape(-1) for m in gen_motions]
            cos_sims = []
            for a in range(len(flat)):
                for b in range(a+1, len(flat)):
                    cs = np.dot(flat[a], flat[b]) / (np.linalg.norm(flat[a]) * np.linalg.norm(flat[b]) + 1e-8)
                    cos_sims.append(cs)
            diversity = 1.0 - float(np.mean(cos_sims))
        else:
            diversity = 0.0

        skel_result = {
            'skeleton': skel_name,
            'n_joints': nj,
            'n_real': len(real_files),
            'n_generated': args.n_samples,
            'real_avg': real_avg,
            'gen_avg': gen_avg,
            'diversity': diversity,
            'gen_vs_real_ratio': {
                k: gen_avg[k] / (real_avg[k] + 1e-8) for k in gen_avg
            } if real_avg else {},
        }
        all_results[skel_name] = skel_result

        print(f"  Generated avg: {gen_avg}")
        print(f"  Gen/Real ratio: {skel_result.get('gen_vs_real_ratio', 'N/A')}")
        print(f"  Diversity: {diversity:.4f}")

        # Save generated motions
        save_path = pjoin(args.out_dir, f'{skel_name}_generated.npy')
        np.save(save_path, np.stack(gen_motions))
        print(f"  Saved → {save_path}")

    # Summary
    print(f"\n{'='*60}")
    print("GENERATION QUALITY SUMMARY")
    print(f"{'='*60}")
    for skel_name, r in all_results.items():
        ratio = r.get('gen_vs_real_ratio', {})
        pos_r = ratio.get('pos_range', 0)
        vel_r = ratio.get('vel_magnitude', 0)
        div = r['diversity']
        nj = r['n_joints']
        print(f"  {skel_name:12s} ({nj:2d}j): pos_ratio={pos_r:.2f} "
              f"vel_ratio={vel_r:.2f} diversity={div:.3f}")

    print("\nInterpretation:")
    print("  pos_ratio ~1.0 = generated motion has similar spatial range to real")
    print("  vel_ratio ~1.0 = generated velocity similar to real")
    print("  diversity >0.2 = samples are diverse (not collapsed)")
    print("  ALL degenerate if pos_range ≈ 0 and vel_magnitude ≈ 0")

    out_path = pjoin(args.out_dir, 'summary.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved summary → {out_path}")


if __name__ == '__main__':
    main()
