"""Probe C: Gauge-violation counterfactual via bilateral relabeling.

Tests whether Idea K's output changes under a semantically-neutral
bilateral relabeling ψ of the source skeleton. For each pair:
  1. Run pipeline normally → output M'
  2. Swap L↔R contact groups in source → run pipeline → output M''
  3. gauge_violation = ‖M' - M''‖_L2 per frame, averaged

A gauge-invariant method should produce identical outputs (distance ≈ 0).
A gauge-sensitive method produces different outputs (high distance).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.skel_blind.invariant_dataset import TEST_SKELETONS

OUT_DIR = PROJECT_ROOT / 'eval/results/probe_c'
MANIFEST = PROJECT_ROOT / 'eval/benchmark_paired/pairs/manifest.json'

BILATERAL_SWAPS = [
    ('LF', 'RF'), ('LH', 'RH'), ('LW', 'RW'),
    ('L', 'R'),
    ('claw_L', 'claw_R'),
    ('L1', 'R1'), ('L2', 'R2'), ('L3', 'R3'), ('L4', 'R4'),
]


def swap_contact_groups(cg_skel: dict) -> dict:
    """Swap L↔R contact group names. Returns new dict with swapped keys."""
    swapped = {}
    swap_map = {}
    for a, b in BILATERAL_SWAPS:
        swap_map[a] = b
        swap_map[b] = a

    for name, joints in cg_skel.items():
        new_name = swap_map.get(name, name)
        swapped[new_name] = joints
    return swapped


def frame_l2(motion_a: np.ndarray, motion_b: np.ndarray) -> float:
    """Mean per-frame L2 distance between two [T, J, D] motions.
    Handles different lengths by truncating to min(T_a, T_b)."""
    T = min(motion_a.shape[0], motion_b.shape[0])
    a = motion_a[:T].reshape(T, -1)
    b = motion_b[:T].reshape(T, -1)
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


def split_category(skel_a, skel_b):
    a_test = skel_a in TEST_SKELETONS
    b_test = skel_b in TEST_SKELETONS
    if a_test and b_test:
        return "test_test"
    elif a_test or b_test:
        return "mixed"
    return "train_train"


def bootstrap_ci(values, n_boot=1000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return (0.0, 0.0, 0.0)
    means = np.array([rng.choice(arr, size=n, replace=True).mean()
                      for _ in range(n_boot)])
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return (round(float(lo), 4), round(float(arr.mean()), 4), round(float(hi), 4))


def run_pipeline_once(src_fname, src_skel, tgt_skel, cond,
                      contact_groups, motion_dir, device,
                      extract_quotient, solve_ik,
                      theta_to_motion_13dim, anytop_project,
                      build_q_star, build_contact_mask_tj,
                      skip_stage3=False):
    """Run the full Idea K pipeline once and return the output motion."""
    q_src = extract_quotient(src_fname, cond[src_skel],
                             contact_groups=contact_groups,
                             motion_dir=motion_dir)
    q_star = build_q_star(q_src, src_skel, tgt_skel,
                          contact_groups, cond)

    ik_out = solve_ik(q_star, cond[tgt_skel],
                      contact_groups[tgt_skel],
                      n_iters=400, verbose=False,
                      device=device)

    try:
        motion_13 = theta_to_motion_13dim(
            ik_out['theta'], ik_out['root_pos'],
            ik_out['positions'],
            tgt_skel, cond,
            contact_groups=contact_groups)
    except Exception:
        motion_13 = theta_to_motion_13dim(
            ik_out['theta'], ik_out['root_pos'],
            ik_out['positions'],
            tgt_skel, cond,
            contact_groups=contact_groups,
            fit_rotations=False)

    if skip_stage3:
        return motion_13

    contact_mask_TJ = build_contact_mask_tj(motion_13)
    com_path_T3 = q_star['com_path'] * q_star['body_scale']
    try:
        proj = anytop_project(
            motion_13, tgt_skel,
            hard_constraints={'contact_positions': contact_mask_TJ,
                              'com_path': com_path_T3},
            t_init=0.3, n_steps=10, device=device)
        return proj['x_refined']
    except Exception:
        return motion_13


def has_bilateral_groups(cg_skel: dict) -> bool:
    """Check if skeleton has at least one bilateral pair in its contact groups."""
    names = set(cg_skel.keys())
    for a, b in BILATERAL_SWAPS:
        if a in names and b in names:
            return True
    return False


def run(max_pairs=200, skip_stage3=False):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(MANIFEST) as f:
        manifest = json.load(f)
    pairs = manifest['pairs'][:max_pairs]
    print(f"Loaded {len(pairs)} pairs from manifest")

    from eval.run_k_pipeline_200pairs import load_assets
    cond, contact_groups, motion_dir = load_assets()

    from eval.quotient_extractor import extract_quotient
    from eval.ik_solver import solve_ik
    from eval.k_pipeline_bridge import theta_to_motion_13dim
    from eval.anytop_projection import anytop_project
    from eval.run_k_pipeline_30pairs import build_q_star, build_contact_mask_tj

    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if skip_stage3:
        device = 'cpu'
    print(f"Device: {device}, skip_stage3: {skip_stage3}")

    per_pair = []
    t_total = time.time()

    for i, p in enumerate(pairs):
        pid = p['pair_id']
        src_skel = p['skel_a']
        tgt_skel = p['skel_b']
        src_fname = p['file_a']
        action = p['action']

        rec = {
            'pair_id': pid, 'action': action,
            'source_skel': src_skel, 'target_skel': tgt_skel,
            'split': split_category(src_skel, tgt_skel),
            'status': 'pending',
        }

        if src_skel not in contact_groups or tgt_skel not in contact_groups:
            rec['status'] = 'skipped_no_cg'
            per_pair.append(rec)
            continue

        if not has_bilateral_groups(contact_groups[src_skel]):
            rec['status'] = 'skipped_no_bilateral'
            per_pair.append(rec)
            continue

        t0 = time.time()
        try:
            pipeline_args = dict(
                src_fname=src_fname, src_skel=src_skel, tgt_skel=tgt_skel,
                cond=cond, motion_dir=motion_dir, device=device,
                extract_quotient=extract_quotient, solve_ik=solve_ik,
                theta_to_motion_13dim=theta_to_motion_13dim,
                anytop_project=anytop_project,
                build_q_star=build_q_star,
                build_contact_mask_tj=build_contact_mask_tj,
                skip_stage3=skip_stage3,
            )

            # Run 1: original contact groups
            m_orig = run_pipeline_once(
                contact_groups=contact_groups, **pipeline_args)

            # Run 2: bilateral-swapped contact groups
            cg_swapped = deepcopy(contact_groups)
            cg_swapped[src_skel] = swap_contact_groups(contact_groups[src_skel])
            m_swapped = run_pipeline_once(
                contact_groups=cg_swapped, **pipeline_args)

            rec['gauge_violation'] = frame_l2(m_orig, m_swapped)
            rec['status'] = 'ok'
            rec['pair_runtime'] = float(time.time() - t0)

            if (i + 1) % 10 == 0 or i == 0:
                elapsed = time.time() - t_total
                eta = elapsed / (i + 1) * (len(pairs) - i - 1)
                print(f"  [{i+1}/{len(pairs)}] {action}: {src_skel}→{tgt_skel} "
                      f"gv={rec['gauge_violation']:.4f} "
                      f"(elapsed {elapsed:.0f}s, ETA {eta:.0f}s)")

        except Exception as e:
            rec['status'] = 'failed'
            rec['error'] = str(e) + '\n' + traceback.format_exc(limit=3)
            rec['pair_runtime'] = float(time.time() - t0)
            print(f"  FAILED pair {pid}: {e}")

        per_pair.append(rec)

    total_time = time.time() - t_total

    # Aggregate
    ok = [r for r in per_pair if r['status'] == 'ok']
    gvs = [r['gauge_violation'] for r in ok]
    skipped_bilateral = sum(1 for r in per_pair if r['status'] == 'skipped_no_bilateral')
    skipped_cg = sum(1 for r in per_pair if r['status'] == 'skipped_no_cg')
    failed = sum(1 for r in per_pair if r['status'] == 'failed')

    print(f"\n{'='*60}")
    print(f"PROBE C — GAUGE VIOLATION (bilateral relabeling)")
    print(f"{'='*60}")
    print(f"Total: {len(per_pair)}, OK: {len(ok)}, "
          f"Skipped (no bilateral): {skipped_bilateral}, "
          f"Skipped (no CG): {skipped_cg}, Failed: {failed}")
    print(f"Time: {total_time:.1f}s")

    if gvs:
        ci = bootstrap_ci(gvs)
        print(f"\nGauge violation distance: {ci[1]:.4f} [{ci[0]:.4f}, {ci[2]:.4f}]")
        print(f"  Median: {np.median(gvs):.4f}")
        print(f"  Max: {max(gvs):.4f}")
        print(f"  Pairs with GV < 0.01: {sum(1 for v in gvs if v < 0.01)}/{len(gvs)}")

    # By split
    by_split = {}
    for cat in ["train_train", "mixed", "test_test"]:
        group = [r for r in ok if r['split'] == cat]
        if group:
            vals = [r['gauge_violation'] for r in group]
            by_split[cat] = {'n': len(group), 'gauge_violation': bootstrap_ci(vals)}
            print(f"  {cat} (n={len(group)}): GV={by_split[cat]['gauge_violation'][1]:.4f}")

    results = {
        'method': 'Idea_K',
        'probe': 'C_gauge_violation',
        'skip_stage3': skip_stage3,
        'n_pairs': len(per_pair),
        'n_ok': len(ok),
        'n_skipped_bilateral': skipped_bilateral,
        'n_skipped_cg': skipped_cg,
        'n_failed': failed,
        'total_time_sec': total_time,
        'overall': {
            'n': len(ok),
            'gauge_violation': bootstrap_ci(gvs) if gvs else None,
            'median': float(np.median(gvs)) if gvs else None,
            'max': float(max(gvs)) if gvs else None,
        },
        'by_split': by_split,
        'per_pair': per_pair,
    }

    out_path = OUT_DIR / 'probe_c_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_pairs', type=int, default=200)
    parser.add_argument('--skip_stage3', action='store_true',
                        help='CPU-only: skip AnyTop projection (Stage 3)')
    args = parser.parse_args()
    run(max_pairs=args.max_pairs, skip_stage3=args.skip_stage3)
