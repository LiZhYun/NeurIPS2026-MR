"""Gauge-theorem empirical validation: latent isotropy (P1) + cross-seed Procrustes (P2).

P1 (isotropy): For a single seed/method, compute the latent covariance eigenvalue
    spectrum. thm:gauge predicts marginal-matching latent training produces
    approximately isotropic latents. Test: spectral flatness (geomean / arithmean
    of eigenvalues, in [0, 1]; 1 = perfectly isotropic). Effective rank:
    exp(-sum(p_i log p_i)) for normalized p_i = lambda_i / sum lambda_j.

P2 (Procrustes alignment): For two seeds of the same method, compute the optimal
    rotation R* aligning Z_a to Z_b. thm:gauge predicts the residual after
    rotation is small relative to the unaligned residual. Report:
        residual_aligned / residual_unaligned  (smaller = stronger gauge equivalence)

Usage:
    python /tmp/gauge_analysis.py --latent_dirs DIR1 DIR2 ... --names NAME1 NAME2 ...
    e.g.,
    python /tmp/gauge_analysis.py \
        --latent_dirs /tmp/ace_latents/no_ladv_seed42 /tmp/ace_latents/no_ladv_seed43 /tmp/ace_latents/no_ladv_seed44 \
        --names s42 s43 s44

Saves: <output>.json with per-seed isotropy stats + pairwise Procrustes residuals.
"""
import argparse
import json
import sys
from pathlib import Path
from itertools import combinations

import numpy as np


def load_latents(latent_dir):
    """Load all z_query_*.npy from dir, plus meta.json. Returns dict {tgt_skel: list of (qid, latent)}."""
    latent_dir = Path(latent_dir)
    meta = json.loads((latent_dir / 'meta.json').read_text())
    by_skel = {}
    for entry in meta:
        qid = entry['query_id']
        tgt = entry['tgt_skel']
        z = np.load(latent_dir / f'z_query_{qid:04d}.npy')  # [T_tokens, codebook_dim]
        by_skel.setdefault(tgt, []).append((qid, z))
    return by_skel


def isotropy_stats(Z):
    """Z: [N, d]. Returns dict with eigenvalues, spectral flatness, effective rank."""
    Z_centered = Z - Z.mean(axis=0, keepdims=True)
    N, d = Z_centered.shape
    if N < d + 1:
        # Underdetermined — need at least d+1 samples for full-rank covariance.
        # Fall back to N-rank effective spectrum.
        cov = Z_centered.T @ Z_centered / max(N - 1, 1)
    else:
        cov = Z_centered.T @ Z_centered / (N - 1)
    eigs = np.linalg.eigvalsh(cov)  # ascending
    eigs = np.maximum(eigs, 0.0)  # clip tiny negatives from numerical noise
    eigs = np.sort(eigs)[::-1]  # descending
    eigs_nz = eigs[eigs > 1e-12]
    if len(eigs_nz) == 0:
        return {'eig_max': 0.0, 'eig_min': 0.0, 'spectral_flatness': float('nan'),
                'effective_rank': 0.0, 'd': int(d), 'N': int(N)}
    geomean = float(np.exp(np.mean(np.log(eigs_nz))))
    arithmean = float(eigs_nz.mean())
    flatness = geomean / arithmean if arithmean > 0 else float('nan')
    p = eigs_nz / eigs_nz.sum()
    eff_rank = float(np.exp(-np.sum(p * np.log(p + 1e-30))))
    return {
        'eig_max': float(eigs_nz[0]),
        'eig_min': float(eigs_nz[-1]),
        'eig_top5_frac': float(eigs_nz[:min(5, len(eigs_nz))].sum() / eigs_nz.sum()),
        'spectral_flatness': float(flatness),
        'effective_rank': eff_rank,
        'd': int(d),
        'N': int(N),
        'eigs_top5': [float(x) for x in eigs_nz[:5]],
    }


def procrustes_residual(Z_a, Z_b):
    """Find optimal rotation R minimising ||Z_b - Z_a @ R||_F.
    Returns dict with residual_aligned, residual_unaligned, fraction_explained, R_eigs."""
    assert Z_a.shape == Z_b.shape
    Mc = Z_a.T @ Z_b  # [d, d]
    U, S, Vt = np.linalg.svd(Mc)
    R = U @ Vt  # [d, d], orthogonal
    # Determinant: this is in O(d), not SO(d) — we accept reflections too,
    # since gauge orbit is O(d).
    Z_a_rot = Z_a @ R
    res_aligned = float(np.sum((Z_b - Z_a_rot) ** 2))
    res_unaligned = float(np.sum((Z_b - Z_a) ** 2))
    res_total = float(np.sum(Z_b ** 2))
    frac_explained = 1.0 - (res_aligned / max(res_unaligned, 1e-12))
    # Mean column correlation after alignment (cosine similarity per row, averaged):
    Z_b_norms = np.linalg.norm(Z_b, axis=1, keepdims=True) + 1e-12
    Z_a_rot_norms = np.linalg.norm(Z_a_rot, axis=1, keepdims=True) + 1e-12
    cosines = (Z_b * Z_a_rot).sum(axis=1) / (Z_b_norms[:, 0] * Z_a_rot_norms[:, 0])
    return {
        'residual_aligned': res_aligned,
        'residual_unaligned': res_unaligned,
        'residual_total_b': res_total,
        'rel_aligned_b': res_aligned / max(res_total, 1e-12),
        'rel_unaligned_b': res_unaligned / max(res_total, 1e-12),
        'fraction_aligned_explained': frac_explained,
        'mean_row_cosine_aligned': float(np.mean(cosines)),
        'singular_values_top5': [float(x) for x in S[:5]],
        'singular_value_sum': float(S.sum()),
        'rotation_trace_over_d': float(np.trace(R) / R.shape[0]),
        'd': int(Z_a.shape[1]),
        'N': int(Z_a.shape[0]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_dirs', nargs='+', required=True, help='Directories with z_query_*.npy + meta.json')
    parser.add_argument('--names', nargs='+', required=True, help='Short names for each dir (same order)')
    parser.add_argument('--out_json', type=str, required=True)
    parser.add_argument('--min_samples_per_skel', type=int, default=20,
                        help='Skip target skels with fewer than this many tokens after concat')
    args = parser.parse_args()
    assert len(args.latent_dirs) == len(args.names)

    print(f"[gauge] Loading {len(args.latent_dirs)} latent dirs...")
    seeds = {}
    for name, d in zip(args.names, args.latent_dirs):
        print(f"  {name}: {d}")
        seeds[name] = load_latents(d)

    # Discover common target skeletons + concat tokens per (seed, tgt_skel).
    common_tgts = set.intersection(*[set(by_skel.keys()) for by_skel in seeds.values()])
    print(f"[gauge] common target skels: {sorted(common_tgts)}")

    per_skel_results = {}
    for tgt in sorted(common_tgts):
        # Collect per-seed concatenated latent matrices for this tgt
        # WITHIN a tgt, both seeds must process the SAME query order so that
        # row i of Z_a and Z_b correspond to the same source motion. Since
        # all seeds load the manifest in the same order, this should hold.
        per_seed_Z = {}
        per_seed_qids = {}
        for name, by_skel in seeds.items():
            entries = by_skel.get(tgt, [])
            if not entries:
                continue
            entries_sorted = sorted(entries, key=lambda x: x[0])
            qids = [qid for qid, _ in entries_sorted]
            zs = np.concatenate([z for _, z in entries_sorted], axis=0)  # [sum_T, d]
            per_seed_Z[name] = zs
            per_seed_qids[name] = qids

        # Sanity: all seeds should have the same query IDs and same N
        ref_qids = next(iter(per_seed_qids.values()))
        ref_N = next(iter(per_seed_Z.values())).shape[0]
        all_match = all(per_seed_qids[k] == ref_qids and per_seed_Z[k].shape[0] == ref_N
                        for k in per_seed_Z)
        if not all_match:
            print(f"  [SKIP {tgt}] mismatched query/sample counts across seeds")
            continue
        if ref_N < args.min_samples_per_skel:
            continue

        d = next(iter(per_seed_Z.values())).shape[1]

        # P1 (isotropy) per seed
        p1 = {name: isotropy_stats(Z) for name, Z in per_seed_Z.items()}

        # P2 (Procrustes) for all pairs
        p2 = {}
        for a, b in combinations(per_seed_Z.keys(), 2):
            p2[f'{a}_vs_{b}'] = procrustes_residual(per_seed_Z[a], per_seed_Z[b])

        per_skel_results[tgt] = {
            'd': d,
            'N': ref_N,
            'P1_isotropy': p1,
            'P2_procrustes_pairs': p2,
        }

    # Aggregate (mean across skels)
    aggregate = {'P1': {}, 'P2': {}}
    for name in args.names:
        flatness_list = [r['P1_isotropy'].get(name, {}).get('spectral_flatness', np.nan)
                          for r in per_skel_results.values() if name in r['P1_isotropy']]
        eff_rank_list = [r['P1_isotropy'].get(name, {}).get('effective_rank', np.nan)
                          for r in per_skel_results.values() if name in r['P1_isotropy']]
        flatness_arr = np.array([x for x in flatness_list if np.isfinite(x)])
        eff_rank_arr = np.array([x for x in eff_rank_list if np.isfinite(x)])
        aggregate['P1'][name] = {
            'spectral_flatness_mean': float(flatness_arr.mean()) if len(flatness_arr) else float('nan'),
            'spectral_flatness_std': float(flatness_arr.std()) if len(flatness_arr) > 1 else float('nan'),
            'effective_rank_mean': float(eff_rank_arr.mean()) if len(eff_rank_arr) else float('nan'),
            'effective_rank_std': float(eff_rank_arr.std()) if len(eff_rank_arr) > 1 else float('nan'),
            'd_typical': next(iter(per_skel_results.values()))['d'] if per_skel_results else None,
            'n_skels': int(len(flatness_arr)),
        }
    for a, b in combinations(args.names, 2):
        key = f'{a}_vs_{b}'
        rel_aligned_list = [r['P2_procrustes_pairs'].get(key, {}).get('rel_aligned_b', np.nan)
                             for r in per_skel_results.values() if key in r['P2_procrustes_pairs']]
        rel_unaligned_list = [r['P2_procrustes_pairs'].get(key, {}).get('rel_unaligned_b', np.nan)
                               for r in per_skel_results.values() if key in r['P2_procrustes_pairs']]
        cos_list = [r['P2_procrustes_pairs'].get(key, {}).get('mean_row_cosine_aligned', np.nan)
                     for r in per_skel_results.values() if key in r['P2_procrustes_pairs']]
        rel_aligned_arr = np.array([x for x in rel_aligned_list if np.isfinite(x)])
        rel_unaligned_arr = np.array([x for x in rel_unaligned_list if np.isfinite(x)])
        cos_arr = np.array([x for x in cos_list if np.isfinite(x)])
        gap = (rel_unaligned_arr - rel_aligned_arr) / np.maximum(rel_unaligned_arr, 1e-12)
        aggregate['P2'][key] = {
            'rel_aligned_mean': float(rel_aligned_arr.mean()) if len(rel_aligned_arr) else float('nan'),
            'rel_unaligned_mean': float(rel_unaligned_arr.mean()) if len(rel_unaligned_arr) else float('nan'),
            'fraction_explained_by_rotation_mean': float(gap.mean()) if len(gap) else float('nan'),
            'mean_row_cosine_aligned_mean': float(cos_arr.mean()) if len(cos_arr) else float('nan'),
            'n_skels': int(len(rel_aligned_arr)),
        }

    out = {
        'methods': args.names,
        'latent_dirs': args.latent_dirs,
        'aggregate': aggregate,
        'per_skel_results': per_skel_results,
    }
    Path(args.out_json).write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[gauge] saved {args.out_json}")
    print(f"\n=== AGGREGATE P1 (isotropy across {len(per_skel_results)} target skels) ===")
    for name, stats in aggregate['P1'].items():
        print(f"  {name}: flatness={stats['spectral_flatness_mean']:.3f} ± {stats['spectral_flatness_std']:.3f}, "
              f"eff_rank={stats['effective_rank_mean']:.1f} ± {stats['effective_rank_std']:.1f} (d={stats['d_typical']})")
    print(f"\n=== AGGREGATE P2 (Procrustes across pairs of seeds) ===")
    for key, stats in aggregate['P2'].items():
        print(f"  {key}: aligned/||Z_b||²={stats['rel_aligned_mean']:.3f}, "
              f"unaligned/||Z_b||²={stats['rel_unaligned_mean']:.3f}, "
              f"frac_explained={stats['fraction_explained_by_rotation_mean']:.3f}, "
              f"row_cos={stats['mean_row_cosine_aligned_mean']:.3f}")


if __name__ == '__main__':
    main()
