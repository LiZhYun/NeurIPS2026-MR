"""E2 Quasi-Paired Benchmark Evaluation.

Runs cross-skeleton motion retargeting on the 49 curated pairs from
eval/benchmark/pairs.csv and computes:
  1. Content similarity (cosine distance in Set Transformer embedding space)
  2. Motion quality metrics (root speed correlation, contact F1, foot sliding)

pairs.csv format:
  tier, source_skeleton, source_motion_file, target_skeleton, notes

The target column is a skeleton NAME (not a motion file). The retargeting
generates a new motion on the target skeleton; there is no ground-truth
target motion (unpaired evaluation).

Usage:
  conda run -n anytop python -m eval.run_benchmark \\
      --model_path save/A1_full_method_bs_4_latentdim_256/model000200000.pt \\
      --evaluator_path eval/checkpoints/st_evaluator.pt \\
      --pairs_csv eval/benchmark/pairs.csv \\
      --out eval/results/benchmark_A1.json
"""

import argparse
import csv
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from eval.motion_metrics import root_speed_correlation, contact_rhythm_f1, foot_sliding


# ─────────────────────────────────────────────────────────────────────────────
# Data / model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(data_npy):
    d = np.load(data_npy, allow_pickle=True).item()
    name_to_idx = {n: i for i, n in enumerate(d['names'])}
    return d, name_to_idx


def get_motion(data, name_to_idx, motion_file, device):
    """Return [1, J_max, 13, T] motion and [1, J_max] mask for a source clip."""
    idx = name_to_idx.get(motion_file)
    if idx is None:
        raise KeyError(f"Motion not found in dataset: {motion_file}")
    mot = data['motions'][idx]   # [J_max, 13, T_max]
    msk = data['masks'][idx]     # [J_max]
    return (torch.tensor(mot[None], dtype=torch.float32, device=device),
            torch.tensor(msk[None], dtype=torch.bool,    device=device))


def load_anytop_model(model_path, device):
    from utils.model_util import create_conditioned_model_and_diffusion, load_model
    import json
    ckpt_dir  = os.path.dirname(model_path)
    with open(os.path.join(ckpt_dir, 'args.json')) as f:
        args_dict = json.load(f)
    from argparse import Namespace
    args = Namespace(**args_dict)
    model, diffusion = create_conditioned_model_and_diffusion(args)
    state = torch.load(model_path, map_location='cpu')
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    load_model(model, state)
    model.to(device)
    model.eval()
    return model, diffusion, args


def load_evaluator(evaluator_path, device):
    from eval.set_transformer_evaluator import SetTransformerEvaluator
    ckpt = torch.load(evaluator_path, map_location='cpu')
    model = SetTransformerEvaluator()
    model.load_state_dict(ckpt['model'])
    model.eval().to(device)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Encode / retarget / embed
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_source(model, motion, mask, offsets, device):
    """Encode source motion → z [1, T', K, d_model].

    offsets: [1, J_max, 3] real rest-pose bone offsets for the source skeleton.
    """
    enc_out = model.encoder(motion, offsets.to(device), mask)
    return enc_out[0] if isinstance(enc_out, tuple) else enc_out


@torch.no_grad()
def retarget_to_skeleton(model, diffusion, z, target_cond, n_steps, device,
                         max_joints, feature_len, n_frames):
    """Sample retargeted motion on target skeleton conditioned on z.

    target_cond: model_kwargs dict built by build_target_condition()
    Returns [1, J_max, 13, T] generated motion.
    """
    mk = {k: (v.to(device) if torch.is_tensor(v) else v)
          for k, v in target_cond['y'].items()}
    mk['z'] = z
    shape = (1, max_joints, feature_len, n_frames)

    sample = diffusion.p_sample_loop(
        model,
        shape,
        clip_denoised=False,
        model_kwargs={'y': mk},
        skip_timesteps=diffusion.num_timesteps - n_steps,
        init_image=None,
        progress=False,
        dump_steps=None,
        noise=None,
        const_noise=False,
        device=device,
    )
    return sample   # [1, J_max, 13, T]


@torch.no_grad()
def embed_motion(evaluator, motion_jft, mask_j):
    """Set Transformer embed → [1, 128] L2-normalized."""
    return evaluator(motion_jft, mask_j)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(model_path, evaluator_path, pairs_csv, data_npy, out_path,
                  device, n_diffusion_steps=100):
    print(f'Loading dataset...')
    data, name_to_idx = load_dataset(data_npy)

    print(f'Loading AnyTop model...')
    model, diffusion, args = load_anytop_model(model_path, device)

    print(f'Loading Set Transformer evaluator...')
    evaluator = load_evaluator(evaluator_path, device)

    # cond_dict has per-skeleton geometry: tpos, offsets, relations, graph_dist, etc.
    from data_loaders.truebones.truebones_utils.get_opt import get_opt
    from model.conditioners import T5Conditioner
    from sample.generate_conditioned import build_target_condition, encode_joints_names
    opt      = get_opt(device)
    cond     = np.load(opt.cond_file, allow_pickle=True).item()
    t5       = T5Conditioner(name='t5-base', finetune=False, word_dropout=0.0,
                             normalize_text=False, device=device)

    with open(pairs_csv) as f:
        pairs = list(csv.DictReader(f))
    print(f'Evaluating {len(pairs)} pairs...')

    results = []
    for row in tqdm(pairs):
        tier     = row['tier']
        src_skel = row['source_skeleton']
        src_file = row['source_motion_file']
        tgt_skel = row['target_skeleton']
        notes    = row['notes']

        try:
            # ── Source: load specific motion clip ──
            src_mot, src_mask = get_motion(data, name_to_idx, src_file, device)

            # ── Build real rest-pose offsets for the source skeleton ──
            base_src = src_skel.split('__')[0]
            raw_offs = cond[base_src]['offsets']           # [J_real, 3]
            J_max = opt.max_joints
            pad = J_max - raw_offs.shape[0]
            padded_offs = np.concatenate([raw_offs, np.zeros((pad, 3))], axis=0)
            src_offsets = torch.tensor(padded_offs[None], dtype=torch.float32)  # [1, J_max, 3]

            # ── Encode source → z ──
            z = encode_source(model, src_mot, src_mask, src_offsets, device)

            # ── Target: build full model_kwargs from cond_dict (skeleton geometry only) ──
            if tgt_skel not in cond:
                results.append({'tier': tier, 'src': src_file, 'tgt': tgt_skel,
                                 'error': f'{tgt_skel} not in cond_dict'})
                continue

            _, tgt_cond = build_target_condition(
                tgt_skel, cond,
                n_frames=args.num_frames,
                temporal_window=args.temporal_window,
                t5_conditioner=t5,
                max_joints=opt.max_joints,
                feature_len=opt.feature_len,
            )

            # ── Retarget ──
            retargeted = retarget_to_skeleton(
                model, diffusion, z, tgt_cond, n_diffusion_steps, device,
                max_joints=opt.max_joints, feature_len=opt.feature_len,
                n_frames=args.num_frames)

            # ── Content similarity via E1 evaluator ──
            # Build simple [1, J_max] bool mask from n_joints (evaluator format)
            n_j = int(tgt_cond['y']['n_joints'][0])
            tgt_bool_mask = torch.zeros(1, opt.max_joints, dtype=torch.bool, device=device)
            tgt_bool_mask[0, :n_j] = True

            emb_src = embed_motion(evaluator, src_mot,    src_mask)       # [1, 128]
            emb_ret = embed_motion(evaluator, retargeted, tgt_bool_mask)  # [1, 128]
            content_sim = float(F.cosine_similarity(emb_src, emb_ret).mean().cpu())

            # ── Motion quality metrics ──
            src_np = src_mot[0].cpu().numpy()     # [J_max, 13, T]
            ret_np = retargeted[0].cpu().numpy()
            rsc  = root_speed_correlation(src_np, ret_np, layout='jft')
            cf1  = contact_rhythm_f1(src_np,      ret_np, layout='jft')
            fs   = foot_sliding(ret_np, layout='jft')

            results.append({
                'tier':          tier,
                'src_skeleton':  src_skel,
                'src_file':      src_file,
                'tgt_skeleton':  tgt_skel,
                'notes':         notes,
                'content_sim':   content_sim,
                'root_speed_r':  rsc,
                'contact_f1':    cf1,
                'foot_sliding':  fs,
            })

        except Exception as e:
            import traceback
            results.append({'tier': tier, 'src': src_file, 'tgt': tgt_skel,
                             'error': str(e), 'tb': traceback.format_exc()})

    # ── Aggregate ──
    summary = {}
    for tier in ['near', 'medium', 'hard']:
        ok = [r for r in results if r.get('tier') == tier and 'content_sim' in r]
        if ok:
            summary[tier] = {
                'n':            len(ok),
                'content_sim':  float(np.mean([r['content_sim']  for r in ok])),
                'root_speed_r': float(np.nanmean([r['root_speed_r'] for r in ok])),
                'contact_f1':   float(np.nanmean([r['contact_f1']   for r in ok])),
                'foot_sliding': float(np.nanmean([r['foot_sliding']  for r in ok])),
            }
    all_ok = [r for r in results if 'content_sim' in r]
    if all_ok:
        summary['overall'] = {
            'n':            len(all_ok),
            'content_sim':  float(np.mean([r['content_sim']  for r in all_ok])),
            'root_speed_r': float(np.nanmean([r['root_speed_r'] for r in all_ok])),
            'contact_f1':   float(np.nanmean([r['contact_f1']   for r in all_ok])),
            'foot_sliding': float(np.nanmean([r['foot_sliding']  for r in all_ok])),
        }

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({'summary': summary, 'per_pair': results}, f, indent=2)

    print('\n=== Benchmark Results ===')
    for tier, vals in summary.items():
        print(f'  {tier:8s}  n={vals["n"]:2d}  '
              f'content_sim={vals["content_sim"]:.3f}  '
              f'root_speed_r={vals["root_speed_r"]:.3f}  '
              f'contact_f1={vals["contact_f1"]:.3f}  '
              f'foot_sliding={vals["foot_sliding"]:.4f}')
    print(f'\nSaved to {out_path}')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',      required=True)
    parser.add_argument('--evaluator_path',  default='eval/checkpoints/st_evaluator.pt')
    parser.add_argument('--pairs_csv',       default='eval/benchmark/pairs.csv')
    parser.add_argument('--data_npy',        default='eval/data/truebones_train.npy')
    parser.add_argument('--out',             default='eval/results/benchmark_A1.json')
    parser.add_argument('--n_steps',         type=int, default=100)
    parser.add_argument('--device',          default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    run_benchmark(
        model_path=args.model_path,
        evaluator_path=args.evaluator_path,
        pairs_csv=args.pairs_csv,
        data_npy=args.data_npy,
        out_path=args.out,
        device=args.device,
        n_diffusion_steps=args.n_steps,
    )
