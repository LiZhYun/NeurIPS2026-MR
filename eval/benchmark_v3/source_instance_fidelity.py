"""Source-Instance Fidelity (SIF) metric.

Measures whether a generator's output preserves source-instance information
beyond what the (skel_b, action_label) cell mean carries. Addresses Codex's
CRITICAL #1 weakness: V5 contrastive AUC measures action-label recovery, not
source-instance retargeting.

Definition. For each (skel_a, skel_b, action_label) triple where there are
n_src >= 2 distinct source clips on skel_a with that action, take the n_src
generator outputs on skel_b. Compute:

  - Pairwise distance matrix D_src[i, j] = d(x_a_i, x_a_j) on the n_src
    source clips, after Procrustes alignment in 3D.
  - Pairwise distance matrix D_out[i, j] = d(x_b_hat_i, x_b_hat_j) on the
    n_src generated outputs.
  - Pearson correlation rho between flatten(D_src) and flatten(D_out)
    (only off-diagonal entries).

Interpretation:

  - rho ~ 1: outputs preserve source variation (source-conditioned)
  - rho ~ 0: outputs are independent of source (source-blind / cell-mean)
  - rho < 0: outputs anti-correlate with source (rare; indicates instability)

Aggregate: per-method macro-rho across triples (skel_a, skel_b, action) with
n_src >= 2; report bootstrap CI by triple resampling.

We additionally report the *output diversity ratio*:
  R = mean(D_out) / mean(D_src)
  R = 0 means all outputs identical (cell-mean collapse).
  R > 0 means outputs vary; R near 1 means outputs preserve source-scale variation.

Usage:
  python -m eval.benchmark_v3.source_instance_fidelity \
      --method_dir eval/results/baselines/anytop_v5/fold_42 \
      --fold 42 --method_name anytop_v5
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_ROOT = PROJECT_ROOT / 'dataset/truebones/zoo/truebones_processed'
MOTION_DIR = DATA_ROOT / 'motions'


def procrustes_3d_distance(a, b):
    """Per-frame Procrustes-aligned mean MSE between motions a, b.
    a, b: [T_a, J, 3], [T_b, J, 3]. We crop to min(T_a, T_b), use only joint positions
    columns 0..3 of the 13-dim per-joint feature.
    """
    if a.shape[1] != b.shape[1]:
        return np.nan
    T = min(a.shape[0], b.shape[0])
    if T < 4:
        return np.nan
    a, b = a[:T], b[:T]
    # Use first 3 dims as joint positions
    if a.shape[-1] >= 3:
        a = a[..., :3]
        b = b[..., :3]
    # Mean center per frame
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    # Per-frame Procrustes
    distances = []
    for t in range(T):
        At, Bt = a[t], b[t]  # [J, 3]
        H = At.T @ Bt  # [3, 3]
        try:
            U, S, Vt = np.linalg.svd(H)
        except np.linalg.LinAlgError:
            continue
        d = np.linalg.det(U @ Vt)
        D = np.diag([1, 1, np.sign(d)])
        R = U @ D @ Vt
        scale = np.trace(np.diag(S) @ D) / max(1e-8, np.sum(At ** 2))
        At_aligned = scale * At @ R
        distances.append(np.mean((At_aligned - Bt) ** 2))
    return float(np.mean(distances)) if distances else np.nan


def crop_to_real_joints(arr, n_joints):
    """If arr is padded to J_max, crop to first n_joints."""
    return arr[:, :n_joints, :]


def load_query_motion(method_dir: Path, query_id: int):
    p = method_dir / f'query_{query_id:04d}.npy'
    if not p.exists():
        return None
    return np.load(p).astype(np.float32)


def compute_sif_per_triple(method_dir: Path, manifest, motion_dir: Path,
                            min_src_clips=2, max_src_clips=8, max_triples=None):
    """For each (skel_a, skel_b, action) triple with >= min_src_clips
    distinct source clips, compute Pearson rho between source-pairwise
    distances and output-pairwise distances.
    """
    queries = manifest['queries']
    # Group queries by (skel_a, skel_b, action)
    by_triple = defaultdict(list)
    for q in queries:
        sa, sb, act = q.get('skel_a'), q.get('skel_b'), q.get('src_action')
        if sa and sb and act:
            by_triple[(sa, sb, act)].append(q)

    triples_used = []
    rhos = []
    diversity_ratios = []

    for (sa, sb, act), qs in sorted(by_triple.items()):
        # Need distinct source clips
        seen_src = {}
        for q in qs:
            sf = q.get('src_fname', '')
            if sf and sf not in seen_src:
                seen_src[sf] = q
        unique_qs = list(seen_src.values())
        if len(unique_qs) < min_src_clips:
            continue
        if len(unique_qs) > max_src_clips:
            unique_qs = unique_qs[:max_src_clips]

        # Load source motions
        src_motions = []
        for q in unique_qs:
            sp = motion_dir / q['src_fname']
            if not sp.exists():
                src_motions.append(None)
            else:
                src_motions.append(np.load(sp).astype(np.float32))
        # Load output motions
        out_motions = []
        for q in unique_qs:
            om = load_query_motion(method_dir, q['query_id'])
            out_motions.append(om)

        # Filter for triples where ALL motions loaded successfully
        valid = [(s, o) for s, o in zip(src_motions, out_motions) if s is not None and o is not None]
        if len(valid) < min_src_clips:
            continue

        n = len(valid)
        # All sources should have same n_joints (same skel_a)
        src_J = valid[0][0].shape[1]
        # All outputs should have same n_joints (same skel_b)
        out_J = valid[0][1].shape[1]

        D_src = np.full((n, n), np.nan)
        D_out = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(i + 1, n):
                D_src[i, j] = D_src[j, i] = procrustes_3d_distance(valid[i][0], valid[j][0])
                D_out[i, j] = D_out[j, i] = procrustes_3d_distance(valid[i][1], valid[j][1])
        # Off-diagonal entries
        mask = ~np.eye(n, dtype=bool) & ~np.isnan(D_src) & ~np.isnan(D_out)
        if mask.sum() < 2:
            continue
        s_vec = D_src[mask]
        o_vec = D_out[mask]
        if s_vec.std() < 1e-8 or o_vec.std() < 1e-8:
            rho = 0.0
        else:
            rho = float(np.corrcoef(s_vec, o_vec)[0, 1])
        if np.isnan(rho):
            continue
        triples_used.append({'skel_a': sa, 'skel_b': sb, 'action': act, 'n': n, 'rho': rho})
        rhos.append(rho)
        # Diversity ratio
        if s_vec.mean() > 1e-8:
            R = float(o_vec.mean() / s_vec.mean())
        else:
            R = 0.0
        diversity_ratios.append(R)
        if max_triples and len(rhos) >= max_triples:
            break

    return triples_used, rhos, diversity_ratios


def bootstrap_ci(values, n_boot=1000, seed=0):
    if not values:
        return None, None, None
    arr = np.array(values, dtype=float)
    rng = np.random.RandomState(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.choice(len(arr), len(arr), replace=True)
        boots.append(arr[idx].mean())
    boots = np.sort(boots)
    return float(arr.mean()), float(boots[int(0.025 * n_boot)]), float(boots[int(0.975 * n_boot)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method_dir', type=str, required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--method_name', type=str, required=True)
    parser.add_argument('--manifest', type=str, default=None,
                        help='Custom manifest path (else V5 default by fold)')
    parser.add_argument('--motion_dir', type=str, default=str(MOTION_DIR))
    parser.add_argument('--min_src_clips', type=int, default=2)
    parser.add_argument('--max_src_clips', type=int, default=6)
    parser.add_argument('--max_triples', type=int, default=0)
    parser.add_argument('--out_path', type=str, default=None)
    args = parser.parse_args()

    method_dir = Path(args.method_dir).resolve()
    if args.manifest:
        manifest_path = Path(args.manifest)
    else:
        manifest_path = PROJECT_ROOT / f'eval/benchmark_v3/queries_v5/fold_{args.fold}/manifest.json'
    manifest = json.load(open(manifest_path))

    print(f"=== SIF: {args.method_name} fold {args.fold} ===")
    print(f"  method_dir: {method_dir}")
    print(f"  manifest: {manifest_path} ({len(manifest['queries'])} queries)")

    t0 = time.time()
    triples, rhos, divs = compute_sif_per_triple(
        method_dir, manifest, Path(args.motion_dir),
        min_src_clips=args.min_src_clips, max_src_clips=args.max_src_clips,
        max_triples=args.max_triples or None,
    )
    print(f"  triples used: {len(triples)} (in {time.time() - t0:.0f}s)")
    if not triples:
        print("  NO eligible triples")
        return

    rho_mean, rho_lo, rho_hi = bootstrap_ci(rhos)
    div_mean, div_lo, div_hi = bootstrap_ci(divs)
    print(f"  Source-Instance Fidelity (rho): {rho_mean:.4f} [{rho_lo:.4f}, {rho_hi:.4f}]")
    print(f"  Output diversity ratio (R):     {div_mean:.4f} [{div_lo:.4f}, {div_hi:.4f}]")

    out = {
        'method': args.method_name,
        'fold': args.fold,
        'method_dir': str(method_dir),
        'n_triples': len(triples),
        'sif_rho_mean': rho_mean,
        'sif_rho_ci': [rho_lo, rho_hi],
        'output_diversity_ratio_mean': div_mean,
        'output_diversity_ratio_ci': [div_lo, div_hi],
        'per_triple': triples,
    }
    out_path = Path(args.out_path) if args.out_path else method_dir / f'sif_metric.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"  saved: {out_path}")


if __name__ == '__main__':
    main()
