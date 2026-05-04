"""Idea G (lightweight): in-context retargeting via prompt-based inference on B1.

Inference-only approach (no fine-tuning) — per plan2.md Idea G instructions and
the 45-min budget, we use the existing B1 behavior-conditioned checkpoint and
inject an in-context TARGET EXAMPLE by concatenating its behavior tokens with
the source's behavior tokens in the cross-attention memory.

For each of 30 eval pairs (src_motion on S → target skeleton T):
  1. Compute source behavior tokens B_src from psi(src_motion) + action(src_motion).
  2. Pick a target-skeleton example clip (prefer same action label).
  3. Compute target-example behavior tokens B_tex from psi(tex) + action(tex).
  4. Concat K-axis: behavior_tokens = [B_src || B_tex]  (2×n_total_tokens tokens).
  5. Sample on target skeleton with these prompts.

Outputs per pair: eval/results/k_compare/in_context_retarget/pair_<id>_<src>_to_<tgt>.npy
                  and metrics.json aggregated over 30 pairs.

Usage:
    conda run -n anytop python -m eval.run_in_context_retarget \
        --ckpt save/B1_scratch_seed42/model000200000.pt
"""
from __future__ import annotations

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import os
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch

ROOT = Path(str(PROJECT_ROOT))
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / 'eval/results/k_compare/in_context_retarget'
OUT_DIR.mkdir(parents=True, exist_ok=True)

EVAL_PAIRS = ROOT / 'idea-stage/eval_pairs.json'
PSI_PATH = ROOT / 'eval/results/effect_cache/psi_all.npy'
META_PATH = ROOT / 'eval/results/effect_cache/clip_metadata.json'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default='save/B1_scratch_seed42/model000200000.pt')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--n_frames', type=int, default=120)
    p.add_argument('--example_action_blend', action='store_true',
                   help='Use source action label on target-example tokens too')
    return p.parse_args()


def load_behavior_model(ckpt_path, device):
    from model.anytop_behavior import AnyTopBehavior
    from utils.model_util import create_gaussian_diffusion
    from data_loaders.truebones.truebones_utils.get_opt import get_opt

    args_path = os.path.join(os.path.dirname(ckpt_path), 'args.json')
    with open(args_path) as f:
        args = json.load(f)

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
def compute_behavior_tokens(model, psi_arr, action_label, device):
    """Compute normalized behavior tokens for one psi/action — matches forward() norm."""
    psi_t = torch.tensor(psi_arr[None, ...], dtype=torch.float32, device=device)
    action_t = torch.tensor([action_label], dtype=torch.long, device=device)
    tokens = model.behavior_recognizer(psi_t, action_t)  # [1, n_total_tokens, D]
    tokens_4d = tokens.unsqueeze(1)  # [1, 1, n_total_tokens, D]
    # L2 normalize as in forward()
    token_norms = tokens_4d.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    tokens_4d = tokens_4d / token_norms * model.z_norm_target
    return tokens_4d  # [1, 1, K, D]


@torch.no_grad()
def generate_with_tokens(model, diffusion, y, behavior_tokens, max_joints, feature_len, n_frames, device):
    """Custom p_sample loop that uses pre-computed behavior tokens.

    To pass behavior_tokens directly to the model while bypassing the recognizer,
    we stash them in y['behavior_tokens'], which AnyTopBehavior.forward() reads.
    """
    y = {k: v for k, v in y.items()}
    y['behavior_tokens'] = behavior_tokens  # bypass recognizer

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
    from train.train_behavior import ACTION_TO_IDX

    fixseed(args.seed)
    dist_util.setup_dist(args.device)
    device = dist_util.dev()

    print(f"Loading behavior model from {args.ckpt}")
    model, diffusion, model_args = load_behavior_model(args.ckpt, device)
    print(f"  n_total_tokens: {model.n_total_tokens} (= {model_args.get('n_behavior_tokens', 8)} psi + 1 action)")

    opt = get_opt(args.device)
    cond_dict = np.load(opt.cond_file, allow_pickle=True).item()
    t5 = T5Conditioner(name=model_args.get('t5_name', 't5-base'),
                       finetune=False, word_dropout=0.0, normalize_text=False, device='cuda')

    psi_all = np.load(PSI_PATH)
    with open(META_PATH) as f:
        metadata = json.load(f)
    fname_to_idx = {m['fname']: i for i, m in enumerate(metadata)}
    fname_to_action = {m['fname']: ACTION_TO_IDX.get(m['coarse_label'], ACTION_TO_IDX['other'])
                       for m in metadata}

    # Build target-example index per skeleton (cached by action label)
    by_skel = {}
    for m in metadata:
        by_skel.setdefault(m['skeleton'], []).append(m)

    # Target-skel cache for y dict
    y_cache = {}
    def get_y(skel):
        if skel not in y_cache:
            y_cache[skel] = build_target_y(skel, cond_dict, opt, t5, args.n_frames, device)
        return y_cache[skel]

    with open(EVAL_PAIRS) as f:
        eval_set = json.load(f)
    pairs = eval_set['pairs']

    per_pair = []
    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    for p in pairs:
        pid = p['pair_id']
        src_fname = p['source_fname']
        src_skel = p['source_skel']
        src_label = p['source_label']
        tgt_skel = p['target_skel']

        # Source behavior
        src_psi_idx = fname_to_idx[src_fname]
        src_action = fname_to_action[src_fname]
        src_psi = psi_all[src_psi_idx]
        src_tokens = compute_behavior_tokens(model, src_psi, src_action, device)  # [1,1,K,D]

        # Target-example clip: prefer same label, else any on target skel
        tgt_clips = by_skel.get(tgt_skel, [])
        same_label = [m for m in tgt_clips if m['coarse_label'] == src_label]
        if same_label:
            tex_meta = same_label[rng.integers(0, len(same_label))]
            tex_source = 'same_label'
        elif tgt_clips:
            tex_meta = tgt_clips[rng.integers(0, len(tgt_clips))]
            tex_source = 'any_label'
        else:
            tex_meta = None
            tex_source = 'none'

        if tex_meta is not None:
            tex_fname = tex_meta['fname']
            tex_psi = psi_all[fname_to_idx[tex_fname]]
            # Use source's action label on tgt-example if blend flag — signals behavior we want
            if args.example_action_blend:
                tex_action = src_action
            else:
                tex_action = fname_to_action[tex_fname]
            tex_tokens = compute_behavior_tokens(model, tex_psi, tex_action, device)
            # Concat along K axis (cross-attention memory axis): doubles the tokens
            combined = torch.cat([src_tokens, tex_tokens], dim=2)  # [1,1,2K,D]
        else:
            tex_fname = None
            combined = src_tokens  # fall back

        # Generate on target skeleton
        target_y, n_joints, mean, std = get_y(tgt_skel)

        sample = generate_with_tokens(
            model, diffusion, target_y, combined,
            opt.max_joints, opt.feature_len, args.n_frames, device)

        motion_norm = sample[0, :n_joints].cpu().permute(2, 0, 1).numpy()
        motion_denorm = motion_norm * (std[:n_joints] + 1e-6) + mean[:n_joints]

        out_path = OUT_DIR / f'pair_{pid:02d}_{src_skel}_to_{tgt_skel}.npy'
        np.save(out_path, motion_denorm.astype(np.float32))

        per_pair.append({
            'pair_id': pid,
            'src_fname': src_fname,
            'src_skel': src_skel,
            'src_label': src_label,
            'tgt_skel': tgt_skel,
            'tgt_example_fname': tex_fname,
            'tgt_example_source': tex_source,
            'n_joints': int(n_joints),
            'out_path': str(out_path.relative_to(ROOT)),
            'family_gap': p['family_gap'],
            'support_same_label': p['support_same_label'],
            'status': 'ok',
        })
        print(f"  pair_{pid:02d}: {src_skel}.{src_label} + {tex_source}({tex_fname}) -> {tgt_skel}  ok  ({n_joints} joints)")

    total_time = time.time() - t0
    metrics = {
        'method': 'in_context_retarget',
        'variant': 'concat_behavior_tokens_B1_inference_only',
        'ckpt': args.ckpt,
        'total_runtime_sec': total_time,
        'n_pairs': len(pairs),
        'n_ok': sum(1 for r in per_pair if r['status'] == 'ok'),
        'per_pair': per_pair,
    }
    (OUT_DIR / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(f"\n=== DONE: total {total_time:.1f}s  n_ok={metrics['n_ok']}/{len(pairs)} ===")
    print(f"Saved to {OUT_DIR}")


if __name__ == '__main__':
    main()
