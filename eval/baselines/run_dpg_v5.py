"""DPG (Direct Paired Generative) inference on V5 benchmark.

Loads a trained DPG checkpoint and generates target motions for each query
via Euler-method flow matching integration.

Usage:
  python -m eval.baselines.run_dpg_v5 --folds 42 43 \
      --ckpt save/dpg/dpg_v1/final.pt --out_tag dpg_v1 --n_steps 20
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loaders.truebones.truebones_utils.param_utils import DATASET_DIR
from eval.benchmark_v3.action_taxonomy import (
    parse_action_from_filename, action_to_cluster, ACTION_CLUSTERS,
)
from model.dpg.dpg_model import DPGModel

DATA_ROOT = Path(DATASET_DIR)
MOTION_DIR = DATA_ROOT / 'motions'
COND_PATH = DATA_ROOT / 'cond.npy'
QUERIES_V5_DIR = PROJECT_ROOT / 'eval/benchmark_v3/queries_v5'
SAVE_ROOT = PROJECT_ROOT / 'save/dpg'

CLUSTERS = sorted(ACTION_CLUSTERS.keys())


def load_model_from_ckpt(ckpt_path: str, device: torch.device):
    ckpt_dir = Path(ckpt_path).parent
    args_json = ckpt_dir / 'args.json'
    args_dict = json.load(open(args_json))
    model = DPGModel(
        d_model=args_dict['d_model'],
        n_layers_src=args_dict['n_layers_src'],
        n_layers_gen=args_dict['n_layers_gen'],
        n_heads=args_dict['n_heads'],
        max_J=args_dict['max_J'],
        max_T=args_dict['max_T'],
        n_skels=args_dict['n_skels'],
        n_exact_actions=args_dict['n_exact_actions'],
        n_clusters=args_dict['n_clusters'],
    ).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state['model'] if 'model' in state else state)
    model.eval()
    return model, args_dict


def pad_motion(motion: np.ndarray, max_T: int, max_J: int):
    T0, J0, _ = motion.shape
    if T0 > max_T: motion = motion[:max_T]; T0 = max_T
    if J0 > max_J: motion = motion[:, :max_J]; J0 = max_J
    pad_T = max_T - T0; pad_J = max_J - J0
    m_pad = np.zeros((max_T, max_J, 13), dtype=np.float32)
    m_pad[:T0, :J0] = motion
    j_mask = np.zeros(max_J, dtype=bool); j_mask[:J0] = True
    t_mask = np.zeros(max_T, dtype=bool); t_mask[:T0] = True
    return m_pad, j_mask, t_mask, T0, J0


@torch.no_grad()
def generate_one(model, args_dict, src_motion: np.ndarray, src_action: str,
                 tgt_skel: str, tgt_T: int, tgt_J: int,
                 skels_sorted: list, exact_to_idx: dict,
                 device: torch.device, n_steps: int = 20) -> np.ndarray:
    """Generate target motion via Euler ODE integration.

    Returns: target_motion [tgt_T, tgt_J, 13]
    """
    max_T = args_dict['max_T']; max_J = args_dict['max_J']
    src_pad, src_jmask, src_tmask, _, _ = pad_motion(src_motion, max_T, max_J)
    src_motion_t = torch.from_numpy(src_pad).unsqueeze(0).to(device)
    src_jmask_t = torch.from_numpy(src_jmask).unsqueeze(0).to(device)
    src_tmask_t = torch.from_numpy(src_tmask).unsqueeze(0).to(device)

    # Action idx (or 0 if unknown)
    aid = exact_to_idx.get(src_action, 0)
    aid_t = torch.tensor([aid], dtype=torch.long, device=device)
    skel_id = skels_sorted.index(tgt_skel) if tgt_skel in skels_sorted else 0
    sid_t = torch.tensor([skel_id], dtype=torch.long, device=device)

    tgt_T_eff = min(tgt_T, max_T)
    tgt_J_eff = min(tgt_J, max_J)
    tgt_jmask = torch.zeros(1, max_J, dtype=torch.bool, device=device)
    tgt_jmask[:, :tgt_J_eff] = True
    tgt_tmask = torch.zeros(1, max_T, dtype=torch.bool, device=device)
    tgt_tmask[:, :tgt_T_eff] = True

    # Encode source ONCE
    src_tokens, _ = model.encode(src_motion_t, src_jmask_t, src_tmask_t)

    # Initialize z_0 from noise
    z = torch.randn(1, max_T, max_J, 13, device=device)
    # Mask invalid slots to 0
    mask_zj = tgt_jmask.unsqueeze(1).unsqueeze(-1).float()
    mask_zt = tgt_tmask.unsqueeze(-1).unsqueeze(-1).float()
    z = z * mask_zj * mask_zt

    # Euler integration: t in [0, 1], n_steps
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t_now = (i + 0.5) / n_steps  # midpoint
        t_b = torch.tensor([t_now], device=device)
        v = model.generate(z, t_b, src_tokens, src_tmask_t,
                            tgt_jmask, tgt_tmask, aid_t, sid_t)
        z = z + dt * v
        z = z * mask_zj * mask_zt

    # Crop to tgt_T_eff, tgt_J_eff
    out = z[0, :tgt_T_eff, :tgt_J_eff, :].cpu().numpy().astype(np.float32)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', nargs='+', type=int, default=[42, 43])
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--out_tag', type=str, default='dpg_v1')
    parser.add_argument('--n_steps', type=int, default=20)
    parser.add_argument('--max_queries', type=int, default=10000)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Loading ckpt: {args.ckpt}")
    model, args_dict = load_model_from_ckpt(args.ckpt, device)
    skels_sorted = args_dict['skels_sorted']
    exact_actions = args_dict['exact_actions']
    exact_to_idx = {a: i for i, a in enumerate(exact_actions)}
    print(f"Model: n_skels={args_dict['n_skels']}, n_actions={len(exact_actions)}")

    cond_dict = np.load(COND_PATH, allow_pickle=True).item()

    for fold in args.folds:
        manifest = json.load(open(QUERIES_V5_DIR / f'fold_{fold}/manifest.json'))
        out_dir = SAVE_ROOT / args.out_tag / f'fold_{fold}'
        out_dir.mkdir(parents=True, exist_ok=True)

        n_done = n_failed = 0
        per_query = []
        t0 = time.time()
        for i, q in enumerate(manifest['queries'][:args.max_queries]):
            qid = q['query_id']
            skel_b = q['skel_b']
            try:
                src_motion = np.load(MOTION_DIR / q['src_fname']).astype(np.float32)
                # Target length: median of positives_cluster, else src_T
                pos_T = [p['T'] for p in q.get('positives_cluster', [])]
                tgt_T = int(np.median(pos_T)) if pos_T else q.get('src_T', 100)
                tgt_J = len(cond_dict[skel_b]['parents'])
                out = generate_one(
                    model, args_dict, src_motion, q['src_action'],
                    skel_b, tgt_T, tgt_J,
                    skels_sorted, exact_to_idx, device, n_steps=args.n_steps)
                np.save(out_dir / f'query_{qid:04d}.npy', out)
                per_query.append({
                    'query_id': qid, 'status': 'ok',
                    'skel_b': skel_b, 'tgt_T': tgt_T, 'tgt_J': tgt_J,
                    'src_action': q['src_action'],
                    'action_in_vocab': q['src_action'] in exact_to_idx,
                })
                n_done += 1
            except Exception as e:
                import traceback
                tb = traceback.format_exc(limit=2)
                print(f"  q{qid} FAILED: {e}\n{tb}")
                per_query.append({'query_id': qid, 'status': 'failed', 'error': str(e)})
                n_failed += 1

            if (i + 1) % 25 == 0 or i == 0:
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (len(manifest['queries']) - i - 1)
                print(f"  fold {fold} [{i+1}/{len(manifest['queries'])}] "
                      f"elapsed {elapsed:.0f}s, ETA {eta:.0f}s, ok={n_done}")

        with open(out_dir / 'metrics.json', 'w') as f:
            json.dump({
                'method': args.out_tag, 'fold': fold, 'ckpt': args.ckpt,
                'n_steps': args.n_steps,
                'n_done': n_done, 'n_failed': n_failed,
                'per_query': per_query,
            }, f, indent=2)
        print(f"\nFold {fold}: {n_done} ok, {n_failed} failed.")


if __name__ == '__main__':
    main()
