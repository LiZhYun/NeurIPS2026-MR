"""Motion-Inversion V-mode ablation on 3 pairs.

Runs the trained Motion-Inversion model with:
  - differential-V (Motion-V, the paper's recipe) → default
  - plain-V (ablation: V_in used directly, no frame-difference)

Output: eval/results/k_compare/motion_inversion_plainv/ (subset of 3 pairs)
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

from eval.run_motion_inversion_30pairs import load_model

EVAL_PAIRS = ROOT / 'idea-stage/eval_pairs.json'
PSI_PATH = ROOT / 'eval/results/effect_cache/psi_all.npy'
META_PATH = ROOT / 'eval/results/effect_cache/clip_metadata.json'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--b1_ckpt', default='save/B1_scratch_seed42/model000200000.pt')
    p.add_argument('--mi_ckpt', default='save/B1_motion_inversion/lora_000500.pt')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--n_frames', type=int, default=120)
    p.add_argument('--lora_rank', type=int, default=16)
    p.add_argument('--lora_alpha', type=int, default=32)
    p.add_argument('--pair_ids', nargs='+', type=int, default=[0, 10, 19],
                   help='Run on a subset of pair_ids for the V-mode ablation.')
    p.add_argument('--out_dir', default='eval/results/k_compare/motion_inversion_plainv')
    return p.parse_args()


def main():
    args = parse_args()
    from utils.fixseed import fixseed
    from utils import dist_util
    from data_loaders.truebones.truebones_utils.get_opt import get_opt
    from model.conditioners import T5Conditioner
    from train.train_behavior import ACTION_TO_IDX
    from eval.run_in_context_retarget import (
        build_target_y, compute_behavior_tokens, generate_with_tokens)

    fixseed(args.seed)
    dist_util.setup_dist(args.device)
    device = dist_util.dev()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build class-side model with use_differential_v=True (module exists; we
    # will flip the per-forward flag instead).
    args.use_differential_v = 1
    args.disable_module = 0
    model, diffusion, model_args = load_model(args, device)
    model.set_mi_enabled(True)

    opt = get_opt(args.device)
    cond_dict = np.load(opt.cond_file, allow_pickle=True).item()
    t5 = T5Conditioner(name=model_args.get('t5_name', 't5-base'),
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
    selected = [p for p in eval_set['pairs'] if p['pair_id'] in args.pair_ids]

    results = {'plain_v': {}, 'diff_v': {}}

    def run_once(mode_name: str, plain_v: bool):
        out_subdir = out_dir / mode_name
        out_subdir.mkdir(parents=True, exist_ok=True)
        model.set_force_plain_v(plain_v)
        per_pair = []
        y_cache = {}
        for p in selected:
            pid = p['pair_id']
            src_fname = p['source_fname']
            src_skel = p['source_skel']
            tgt_skel = p['target_skel']
            src_action = fname_to_action[src_fname]
            src_psi = psi_all[fname_to_idx[src_fname]]
            tokens = compute_behavior_tokens(model, src_psi, src_action, device)

            if tgt_skel not in y_cache:
                y_cache[tgt_skel] = build_target_y(tgt_skel, cond_dict, opt, t5,
                                                    args.n_frames, device)
            target_y, n_joints, mean, std = y_cache[tgt_skel]
            sample = generate_with_tokens(
                model, diffusion, target_y, tokens,
                opt.max_joints, opt.feature_len, args.n_frames, device)
            motion_norm = sample[0, :n_joints].cpu().permute(2, 0, 1).numpy()
            motion_denorm = motion_norm * (std[:n_joints] + 1e-6) + mean[:n_joints]
            out_path = out_subdir / f'pair_{pid:02d}_{src_skel}_to_{tgt_skel}.npy'
            np.save(out_path, motion_denorm.astype(np.float32))
            per_pair.append({'pair_id': pid,
                             'src_fname': src_fname,
                             'src_skel': src_skel,
                             'tgt_skel': tgt_skel,
                             'n_joints': int(n_joints),
                             'out_path': str(out_path.relative_to(ROOT))})
            print(f"  [{mode_name}] pair_{pid:02d}: {src_skel} -> {tgt_skel} ok")
        return per_pair

    print("=== Plain-V (ablation: V used directly, no differential) ===")
    results['plain_v']['per_pair'] = run_once('plain_v', plain_v=True)
    print("=== Differential-V (Motion-V, paper's recipe) ===")
    results['diff_v']['per_pair'] = run_once('diff_v', plain_v=False)

    (out_dir / 'metrics.json').write_text(json.dumps({
        'method': 'motion_inversion_vmode_ablation',
        'pair_ids': args.pair_ids,
        'results': results,
    }, indent=2))
    print(f"Saved ablation outputs to {out_dir}")


if __name__ == '__main__':
    main()
