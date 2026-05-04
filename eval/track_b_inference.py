"""Track B: cross-skeleton inference + behavior preservation evaluation.

Generates target motions from source clips on different skeletons using a trained
behavior-conditioned model. Evaluates with truly external metrics:
  - Caption Retention via TMR or T2M4LVO (locked path)
  - Action transfer accuracy via pretrained classifier
  - TMR Frechet distance for naturalness
  - Source-swap differentiation control
  - Nearest-neighbor retrieval baseline
  - Random-z null baseline

Per refine-logs/effect_program/EMPIRICAL_PLAN.md (post Round 7-8 fixes).

Usage:
    conda run -n anytop python -m eval.track_b_inference --ckpt save/B1_scratch_seed42/model000200000.pt
"""
import os
import json
import time
import argparse
import numpy as np
import torch
from os.path import join as pjoin


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True, help='Trained AnyTopBehavior checkpoint')
    p.add_argument('--n_pairs', type=int, default=200)
    p.add_argument('--n_targets_per_source', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--out_dir', default='eval/results/track_b')
    return p.parse_args()


def load_behavior_model(ckpt_path, device):
    """Load trained AnyTopBehavior model."""
    from model.anytop_behavior import AnyTopBehavior
    from utils.model_util import create_gaussian_diffusion

    args_path = pjoin(os.path.dirname(ckpt_path), 'args.json')
    with open(args_path) as f:
        args = json.load(f)

    from data_loaders.truebones.truebones_utils.get_opt import get_opt
    opt = get_opt(0)

    model = AnyTopBehavior(
        max_joints=opt.max_joints, feature_len=opt.feature_len,
        latent_dim=args['latent_dim'], ff_size=args['latent_dim']*4,
        num_layers=args['layers'], num_heads=4, t5_out_dim=768,
        n_actions=12, n_behavior_tokens=args.get('n_behavior_tokens', 8),
        use_residual=args.get('use_residual', False),
        skip_t5=False, cond_mode='object_type', cond_mask_prob=0.1,
    )
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'])
    model.to(device)
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)

    class DiffArgs:
        noise_schedule = 'cosine'
        sigma_small = True
        diffusion_steps = 100
        lambda_fs = 0.0
        lambda_geo = 0.0
    diffusion = create_gaussian_diffusion(DiffArgs())

    return model, diffusion, args


def build_target_y(skel_name, cond_dict, opt, t5, n_frames, device):
    """Build conditioning y dict for sampling on a target skeleton."""
    from data_loaders.truebones.data.dataset import create_temporal_mask_for_window
    from data_loaders.tensors import create_padded_relation

    info = cond_dict[skel_name]
    max_joints = opt.max_joints
    feature_len = opt.feature_len
    n_joints = len(info['joints_names'])
    mean = info['mean']
    std = info['std'] + 1e-6

    tpos = (info['tpos_first_frame'] - mean) / std
    tpos = np.nan_to_num(tpos)
    tpos_padded = np.zeros((max_joints, feature_len))
    tpos_padded[:n_joints] = tpos
    tpos_t = torch.tensor(tpos_padded).float().unsqueeze(0).to(device)

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

    return {
        'joints_mask':       jmask_5d,
        'mask':              tmask_t,
        'tpos_first_frame':  tpos_t,
        'joints_names_embs': names_t,
        'graph_dist':        gd_t,
        'joints_relations':  jr_t,
        'crop_start_ind':    torch.zeros(1, dtype=torch.long, device=device),
        'n_joints':          torch.tensor([n_joints]),
    }, n_joints, mean, std


@torch.no_grad()
def generate_with_behavior(model, diffusion, y, psi, action_label, max_joints, feature_len, n_frames, device):
    """Sample motion conditioned on behavior (psi + action)."""
    y = {k: v for k, v in y.items()}
    y['psi'] = torch.tensor(psi[None, ...], dtype=torch.float32, device=device)
    y['action_label'] = torch.tensor([action_label], dtype=torch.long, device=device)

    x = torch.randn(1, max_joints, feature_len, n_frames, device=device)
    for t_val in reversed(range(diffusion.num_timesteps)):
        t = torch.tensor([t_val], device=device)
        out = diffusion.p_mean_variance(model, x, t, model_kwargs={'y': y}, clip_denoised=False)
        x = out['mean']
        if t_val > 0:
            noise = torch.randn_like(x)
            x = x + torch.exp(0.5 * out['log_variance']) * noise
    return x


def main():
    args = parse_args()
    from utils.fixseed import fixseed
    from utils import dist_util
    from data_loaders.truebones.truebones_utils.get_opt import get_opt
    from model.conditioners import T5Conditioner

    fixseed(args.seed)
    dist_util.setup_dist(args.device)
    device = dist_util.dev()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading behavior model from {args.ckpt}")
    model, diffusion, model_args = load_behavior_model(args.ckpt, device)
    n_frames = model_args.get('num_frames', 120)

    opt = get_opt(args.device)
    cond_dict = np.load(opt.cond_file, allow_pickle=True).item()
    t5 = T5Conditioner(name=model_args.get('t5_name', 't5-base'),
                       finetune=False, word_dropout=0.0, normalize_text=False, device='cuda')

    psi_all = np.load('eval/results/effect_cache/psi_all.npy')
    with open('eval/results/effect_cache/clip_metadata.json') as f:
        metadata = json.load(f)
    fname_to_psi_idx = {m['fname']: i for i, m in enumerate(metadata)}

    from train.train_behavior import ACTION_TO_IDX
    fname_to_action = {m['fname']: ACTION_TO_IDX.get(m['coarse_label'], ACTION_TO_IDX['other'])
                       for m in metadata}

    skels = sorted(set(m['skeleton'] for m in metadata))
    rng = np.random.default_rng(args.seed)

    # Build pairs: source clip × target skeleton (different from source)
    val_clips = [m for m in metadata if m['split'] == 'val']
    pairs = []
    for _ in range(args.n_pairs):
        src_meta = val_clips[int(rng.integers(0, len(val_clips)))]
        target_skels = [s for s in skels if s != src_meta['skeleton']]
        target = str(rng.choice(target_skels))
        pairs.append((src_meta['fname'], src_meta['skeleton'], target))

    print(f"\nGenerating {len(pairs)} cross-skeleton motions...")
    t0 = time.time()
    generated = []

    for i, (src_fname, src_skel, tgt_skel) in enumerate(pairs):
        psi_idx = fname_to_psi_idx.get(src_fname)
        if psi_idx is None:
            continue
        psi = psi_all[psi_idx]
        action = fname_to_action.get(src_fname, 11)

        target_y, n_joints, mean, std = build_target_y(
            tgt_skel, cond_dict, opt, t5, n_frames, device)

        sample = generate_with_behavior(
            model, diffusion, target_y, psi, action,
            opt.max_joints, opt.feature_len, n_frames, device)

        motion_norm = sample[0, :n_joints].cpu().permute(2, 0, 1).numpy()
        motion_denorm = motion_norm * (std[:n_joints] + 1e-6) + mean[:n_joints]

        generated.append({
            'src_fname': src_fname,
            'src_skel': src_skel,
            'tgt_skel': tgt_skel,
            'target_n_joints': int(n_joints),
            'motion': motion_denorm.tolist(),  # serialize for json
        })

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(pairs) - i - 1) / rate
            print(f"  [{i+1}/{len(pairs)}] {src_skel}→{tgt_skel}, rate={rate:.2f}/s, eta={eta:.0f}s")

    # Save as object array (heterogeneous shapes across skeletons)
    out_path = pjoin(args.out_dir, f'generated_motions.npz')
    motions_obj = np.empty(len(generated), dtype=object)
    for i, g in enumerate(generated):
        motions_obj[i] = np.array(g['motion'], dtype=np.float32)
    pairs_obj = np.array([(p[0], p[1], p[2]) for p in pairs[:len(generated)]], dtype=object)
    np.savez(out_path, pairs=pairs_obj, motions=motions_obj, allow_pickle=True)
    print(f"\nSaved {len(generated)} motions to {out_path}")
    print("Next: run track_b_evaluator.py on these to compute CR / action transfer / Frechet")


if __name__ == '__main__':
    main()
