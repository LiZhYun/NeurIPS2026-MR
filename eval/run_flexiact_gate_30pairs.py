"""Pilot B inference — sample 30 eval pairs with the FlexiAct z-gate on B1.

Loads the base B1 behavior-conditioned checkpoint, swaps in the flexigate model
class, loads the trained gate weights, then runs DDPM sampling for each of the
30 canonical eval pairs. Mirrors run_vmc_temporal_lora_30pairs.py exactly
(target-y builder, behavior-token computation, generate_with_tokens) so the
outputs are apples-to-apples with K / K_ikprior / vmc_temporal_lora.

Output:
    eval/results/k_compare/flexiact_gate/pair_<id>_<src>_to_<tgt>.npy
    eval/results/k_compare/flexiact_gate/metrics.json
"""
from __future__ import annotations

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch

ROOT = Path(str(PROJECT_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.anytop_behavior_flexigate import AnyTopBehaviorFlexiGate
from eval.run_in_context_retarget import (
    build_target_y, compute_behavior_tokens, generate_with_tokens,
)

OUT_DIR = ROOT / 'eval/results/k_compare/flexiact_gate'
OUT_DIR.mkdir(parents=True, exist_ok=True)

EVAL_PAIRS = ROOT / 'idea-stage/eval_pairs.json'
PSI_PATH = ROOT / 'eval/results/effect_cache/psi_all.npy'
META_PATH = ROOT / 'eval/results/effect_cache/clip_metadata.json'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--b1_ckpt',
                   default='save/B1_scratch_seed42/model000200000.pt')
    p.add_argument('--gate_ckpt',
                   default='save/B1_flexiact_gate/model_000200.pt')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--n_frames', type=int, default=120)
    return p.parse_args()


def main():
    args = parse_args()

    from utils.fixseed import fixseed
    from utils import dist_util
    from data_loaders.truebones.truebones_utils.get_opt import get_opt
    from model.conditioners import T5Conditioner
    from utils.model_util import create_gaussian_diffusion
    from train.train_behavior import ACTION_TO_IDX

    fixseed(args.seed)
    dist_util.setup_dist(args.device)
    device = dist_util.dev()

    b1_args_path = os.path.join(os.path.dirname(args.b1_ckpt), 'args.json')
    with open(b1_args_path) as f:
        b1_args = json.load(f)

    opt = get_opt(args.device)
    model = AnyTopBehaviorFlexiGate(
        max_joints=opt.max_joints, feature_len=opt.feature_len,
        latent_dim=b1_args['latent_dim'], ff_size=b1_args['latent_dim'] * 4,
        num_layers=b1_args['layers'], num_heads=4, t5_out_dim=768,
        n_actions=12, n_behavior_tokens=b1_args.get('n_behavior_tokens', 8),
        use_residual=b1_args.get('use_residual', False),
        skip_t5=False, cond_mode='object_type', cond_mask_prob=0.1,
    )
    state = torch.load(args.b1_ckpt, map_location='cpu', weights_only=False)
    state_dict = state['model'] if isinstance(state, dict) and 'model' in state else state
    model.load_state_dict(state_dict, strict=False)

    gate_ckpt = torch.load(args.gate_ckpt, map_location='cpu', weights_only=False)
    model.gate.load_state_dict(gate_ckpt['gate'])
    print(f"Loaded gate weights from {args.gate_ckpt} (step {gate_ckpt.get('step', '?')})")

    model.to(device)
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)

    class DiffArgs:
        noise_schedule = 'cosine'; sigma_small = True; diffusion_steps = 100
        lambda_fs = 0.0; lambda_geo = 0.0
    diffusion = create_gaussian_diffusion(DiffArgs())

    cond_dict = np.load(opt.cond_file, allow_pickle=True).item()
    t5 = T5Conditioner(name=b1_args.get('t5_name', 't5-base'),
                       finetune=False, word_dropout=0.0, normalize_text=False,
                       device='cuda')

    psi_all = np.load(PSI_PATH)
    with open(META_PATH) as f:
        metadata = json.load(f)
    fname_to_idx = {m['fname']: i for i, m in enumerate(metadata)}
    fname_to_action = {m['fname']: ACTION_TO_IDX.get(m['coarse_label'], ACTION_TO_IDX['other'])
                       for m in metadata}

    with open(EVAL_PAIRS) as f:
        eval_set = json.load(f)
    pairs = eval_set['pairs']

    y_cache = {}

    def get_y(skel):
        if skel not in y_cache:
            y_cache[skel] = build_target_y(
                skel, cond_dict, opt, t5, args.n_frames, device)
        return y_cache[skel]

    per_pair = []
    t_total = time.time()
    for p in pairs:
        pid = p['pair_id']
        src_fname = p['source_fname']
        src_skel = p['source_skel']
        tgt_skel = p['target_skel']
        src_action = fname_to_action[src_fname]

        src_psi = psi_all[fname_to_idx[src_fname]]
        tokens = compute_behavior_tokens(model, src_psi, src_action, device)  # [1,1,K,D]

        target_y, n_joints, mean, std = get_y(tgt_skel)
        t0 = time.time()
        sample = generate_with_tokens(
            model, diffusion, target_y, tokens,
            opt.max_joints, opt.feature_len, args.n_frames, device)
        gen_t = time.time() - t0

        motion_norm = sample[0, :n_joints].cpu().permute(2, 0, 1).numpy()
        motion_denorm = motion_norm * (std[:n_joints] + 1e-6) + mean[:n_joints]

        out_path = OUT_DIR / f'pair_{pid:02d}_{src_skel}_to_{tgt_skel}.npy'
        np.save(out_path, motion_denorm.astype(np.float32))

        per_pair.append({
            'pair_id': pid, 'src_fname': src_fname, 'src_skel': src_skel,
            'tgt_skel': tgt_skel, 'n_joints': int(n_joints),
            'family_gap': p['family_gap'],
            'support_same_label': p['support_same_label'],
            'out_path': str(out_path.relative_to(ROOT)),
            'gen_time_sec': gen_t, 'status': 'ok',
        })
        print(f"  pair_{pid:02d}: {src_skel} -> {tgt_skel}  ({n_joints} joints, {gen_t:.1f}s)")

    total_time = time.time() - t_total
    metrics = {
        'method': 'flexiact_gate',
        'b1_ckpt': args.b1_ckpt,
        'gate_ckpt': args.gate_ckpt,
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
