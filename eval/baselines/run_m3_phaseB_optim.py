"""M3 Phase B — per-query test-time motion-space optimization.

Per FINAL_PROPOSAL V4 (Codex 7.5/10): retrieval-anchored, AdaMimic
scale-vanishing residual, FAST-style multi-loss objective with KL anchor
to candidate, no learned model — pure inference-time gradient descent.

Pipeline:
  1. Get top-1 candidate from M3 Phase A (eval/baselines/run_m3_physics_optim.py)
  2. Initialize residual Δ = 0, gate s ∈ [0, 1]^T
  3. Compute target Q* from source motion (numpy, frozen reference)
  4. Optimize Δ, s with Adam for N iterations, loss:
       L = λ_Q * ||Q(c + s·Δ) - Q*||²
         + λ_bone * bone_validity(c + s·Δ)
         + λ_smooth * smoothness(c + s·Δ)
         + λ_phys * foot_skate(c + s·Δ)
         + λ_anchor * ||(c + s·Δ) - c||²        (FAST-style KL-flavored anchor in feature space)
         + λ_gate * ||s||_1                      (scale-vanishing prior)
  5. Save final motion as query_NNNN.npy

Usage:
  python -m eval.baselines.run_m3_phaseB_optim --folds 42 \
      --phase_a_dir save/m3/m3_rerank_v1 \
      --out_tag m3B_optim_v1 --n_iter 80
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
from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np

from eval.m3_torch.q_torch_lite import (
    q_torch_lite, q_distance_torch, smoothness_loss_torch,
    bone_validity_loss_torch, foot_skate_loss_torch,
)

DATA_ROOT = Path(DATASET_DIR)
MOTION_DIR = DATA_ROOT / 'motions'
QUERIES_V5_DIR = PROJECT_ROOT / 'eval/benchmark_v3/queries_v5'
COND_PATH = DATA_ROOT / 'cond.npy'
SAVE_ROOT = PROJECT_ROOT / 'save/m3'


def find_foot_indices(cond_b: dict) -> list:
    """Heuristic foot index extraction for skel_b."""
    return list(cond_b.get('foot_indices', cond_b.get('foot', [])))


def optimize_one_query(
    candidate_motion: np.ndarray,   # [T_b, J_b, 13]
    source_q_target: torch.Tensor,  # [24] precomputed
    skel_b: str,
    cond_b: dict,
    device: torch.device,
    n_iter: int = 80,
    lr: float = 1e-2,
    weights: dict = None,
    log_every: int = 0,
) -> tuple[np.ndarray, dict]:
    """Run per-query M3 Phase B optimization.

    Returns: (optimized_motion [T, J, 13], info dict)
    """
    if weights is None:
        weights = {'q': 1.0, 'bone': 0.5, 'smooth': 0.1, 'phys': 0.3,
                   'anchor': 0.2, 'gate': 0.05}

    T, J, _ = candidate_motion.shape
    c = torch.from_numpy(candidate_motion).float().to(device)  # [T, J, 13]

    # Variables: Δ (full motion residual) + s (per-frame gate ∈ [0, 1])
    delta = torch.zeros_like(c, requires_grad=True)
    s_logit = torch.zeros(T, device=device, requires_grad=True)  # sigmoid → s

    # Frozen targets (no grad)
    rest_offsets = torch.from_numpy(cond_b['offsets']).float().to(device)  # [J, 3]
    parents = list(cond_b['parents'])
    foot_indices = find_foot_indices(cond_b)

    optimizer = torch.optim.Adam([delta, s_logit], lr=lr)

    init_loss = None
    final_loss = None
    history = []

    for it in range(n_iter):
        optimizer.zero_grad()
        s = torch.sigmoid(s_logit)                       # [T] in [0, 1]
        x = c + s.unsqueeze(-1).unsqueeze(-1) * delta    # broadcast [T,1,1] * [T,J,13]

        # Q-feature loss (the main identifiable signal)
        q_pred = q_torch_lite(x)
        l_q = q_distance_torch(q_pred, source_q_target, mode='cosine')

        # Smoothness on position channels
        l_smooth = smoothness_loss_torch(x)

        # Recover positions for bone-validity + foot-skate (may fail if quat recovery breaks)
        # Use PROXY: directly use position channels [..., 1:, :3] without recovery
        # Bone validity in feature space (proxy): consecutive joint position differences
        positions_proxy = x[..., 1:, :3]  # [T, J-1, 3], skip root (channel 0 has special encoding)

        # Pad back to J slots (joint 0 is root; use zeros for proxy)
        full_positions = torch.cat([
            torch.zeros(T, 1, 3, device=device),
            positions_proxy
        ], dim=1)  # [T, J, 3]

        l_bone = bone_validity_loss_torch(full_positions, parents, rest_offsets)

        # Foot skate (use raw position channels as proxy + soft contact from x[..., 12])
        soft_contact = torch.sigmoid((x[..., 12] - 0.5) * 10.0)  # binarize-ish
        l_phys = foot_skate_loss_torch(full_positions, soft_contact, foot_indices)

        # Anchor: keep optimized motion close to candidate
        l_anchor = (x - c).pow(2).mean()

        # Gate sparsity (encourages s → 0 unless really useful)
        l_gate = s.abs().mean()

        loss = (
            weights['q'] * l_q
            + weights['bone'] * l_bone
            + weights['smooth'] * l_smooth
            + weights['phys'] * l_phys
            + weights['anchor'] * l_anchor
            + weights['gate'] * l_gate
        )

        if init_loss is None:
            init_loss = loss.item()

        loss.backward()
        # Clip gradients to avoid explosions
        torch.nn.utils.clip_grad_norm_([delta, s_logit], max_norm=1.0)
        optimizer.step()

        if log_every > 0 and it % log_every == 0:
            history.append({
                'it': it, 'loss': loss.item(),
                'l_q': l_q.item(), 'l_bone': l_bone.item(),
                'l_smooth': l_smooth.item(), 'l_phys': l_phys.item(),
                'l_anchor': l_anchor.item(), 'l_gate': l_gate.item(),
                's_mean': s.mean().item(), 's_max': s.max().item(),
            })

    final_loss = loss.item()
    with torch.no_grad():
        s = torch.sigmoid(s_logit)
        x_final = (c + s.unsqueeze(-1).unsqueeze(-1) * delta).cpu().numpy()

    info = {
        'n_iter': n_iter, 'init_loss': init_loss, 'final_loss': final_loss,
        'loss_reduction': init_loss - final_loss,
        's_mean': float(s.mean().item()),
        's_max': float(s.max().item()),
        'delta_norm': float(delta.norm().item()),
        'history': history,
    }
    return x_final, info


def precompute_source_q(src_fname: str, device: torch.device) -> torch.Tensor:
    """Compute source motion's Q feature in torch (frozen reference)."""
    src_motion = np.load(MOTION_DIR / src_fname).astype(np.float32)
    src_t = torch.from_numpy(src_motion).to(device)
    with torch.no_grad():
        q = q_torch_lite(src_t)
    return q.detach()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', nargs='+', type=int, default=[42])
    parser.add_argument('--phase_a_dir', type=str,
                        default='save/m3/m3_rerank_v1',
                        help='Phase A output dir (with fold_NN/query_NNNN.npy)')
    parser.add_argument('--out_tag', type=str, default='m3B_optim_v1')
    parser.add_argument('--n_iter', type=int, default=80)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--max_queries', type=int, default=10000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_every', type=int, default=0,
                        help='Log loss every N iterations (0 = no per-iter logging)')
    parser.add_argument('--w_q', type=float, default=1.0)
    parser.add_argument('--w_bone', type=float, default=0.5)
    parser.add_argument('--w_smooth', type=float, default=0.1)
    parser.add_argument('--w_phys', type=float, default=0.3)
    parser.add_argument('--w_anchor', type=float, default=0.2)
    parser.add_argument('--w_gate', type=float, default=0.05)
    args = parser.parse_args()

    weights = {'q': args.w_q, 'bone': args.w_bone, 'smooth': args.w_smooth,
               'phys': args.w_phys, 'anchor': args.w_anchor, 'gate': args.w_gate}
    print(f"M3 Phase B optim. n_iter={args.n_iter}, lr={args.lr}")
    print(f"Weights: {weights}")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    cond_dict = np.load(COND_PATH, allow_pickle=True).item()

    phase_a_root = Path(args.phase_a_dir)
    if not phase_a_root.is_absolute():
        phase_a_root = PROJECT_ROOT / phase_a_root

    for fold in args.folds:
        manifest = json.load(open(QUERIES_V5_DIR / f'fold_{fold}/manifest.json'))
        out_dir = SAVE_ROOT / args.out_tag / f'fold_{fold}'
        out_dir.mkdir(parents=True, exist_ok=True)
        phase_a_fold = phase_a_root / f'fold_{fold}'
        if not phase_a_fold.exists():
            print(f"WARN: Phase A fold dir not found: {phase_a_fold}")
            continue

        n_done = n_failed = 0
        per_query = []
        t0 = time.time()
        for i, q in enumerate(manifest['queries'][:args.max_queries]):
            qid = q['query_id']
            skel_b = q['skel_b']
            src_fname = q['src_fname']

            phase_a_path = phase_a_fold / f'query_{qid:04d}.npy'
            if not phase_a_path.exists():
                n_failed += 1
                per_query.append({'query_id': qid, 'status': 'no_phase_a'})
                continue

            try:
                candidate = np.load(phase_a_path).astype(np.float32)
                src_q = precompute_source_q(src_fname, device)
                cond_b = cond_dict[skel_b]
                # Align candidate joints to cond_b
                n_joints_b = len(cond_b['parents'])
                if candidate.shape[1] > n_joints_b:
                    candidate = candidate[:, :n_joints_b]
                opt_motion, info = optimize_one_query(
                    candidate, src_q, skel_b, cond_b, device,
                    n_iter=args.n_iter, lr=args.lr, weights=weights,
                    log_every=args.log_every,
                )
                np.save(out_dir / f'query_{qid:04d}.npy', opt_motion.astype(np.float32))
                per_query.append({
                    'query_id': qid, 'status': 'ok',
                    'init_loss': info['init_loss'], 'final_loss': info['final_loss'],
                    'loss_reduction': info['loss_reduction'],
                    's_mean': info['s_mean'], 's_max': info['s_max'],
                    'delta_norm': info['delta_norm'],
                })
                n_done += 1
            except Exception as e:
                import traceback
                tb = traceback.format_exc(limit=2)
                print(f"  q{qid} FAILED: {e}\n{tb}")
                per_query.append({'query_id': qid, 'status': 'failed', 'error': str(e)})
                n_failed += 1

            if (i + 1) % 50 == 0 or i == 0:
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (len(manifest['queries']) - i - 1)
                print(f"  fold {fold} [{i+1}/{len(manifest['queries'])}] "
                      f"elapsed {elapsed:.0f}s, ETA {eta:.0f}s, "
                      f"ok={n_done}, failed={n_failed}")

        summary = {
            'method': args.out_tag, 'fold': fold,
            'n_iter': args.n_iter, 'lr': args.lr, 'weights': weights,
            'n_done': n_done, 'n_failed': n_failed,
            'total_time_sec': time.time() - t0,
            'per_query': per_query,
        }
        with open(out_dir / 'metrics.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nFold {fold}: {n_done} ok, {n_failed} failed. Saved to {out_dir}")


if __name__ == '__main__':
    main()
