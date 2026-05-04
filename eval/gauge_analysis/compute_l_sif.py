"""L-SIF: latent-space analogue of Source-Instance Fidelity.

For each SIF triple (skel_a, skel_b, action) with >=2 distinct source clips:
  D_src[i,j] = procrustes-aligned per-frame distance between source motions i,j (in skel_a coords)
  D_lat[i,j] = Frobenius distance between target latents for queries i,j (in skel_b latent)
  L-SIF rho per triple = Pearson(D_src, D_lat)

Reading:
  L-SIF >> 0  : different sources -> different target latents (latent space preserves source variation)
  L-SIF ~~ 0  : different sources -> same / similar target latents (latent collapse)

Combined with standard SIF (output-space):
  L-SIF >> 0 AND SIF ~~ 0  : decoder collapse (latent has source info, decoder drops it)
  L-SIF ~~ 0 AND SIF ~~ 0  : latent collapse (information lost before decoder)
  L-SIF >> 0 AND SIF >> 0  : source-conditional throughout (good case)

Usage:
    python compute_l_sif.py --method ACE-T_s42 --latent_dir /tmp/ace_latents/ace_primary_70 \
        --manifest /home//Codes/Anytop/eval/benchmark_v3/queries_sif_intersection/manifest.json \
        --out_json eval/gauge_analysis/results/lsif_ACE-T_s42.json
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loaders.truebones.truebones_utils.param_utils import DATASET_DIR


# ---------- Procrustes 3D distance (re-implementation, identical to SIF metric) ----------

def procrustes_3d_distance(a, b):
    """a, b: [T, J, 13] motion arrays in same skeleton's coords. Returns scalar distance."""
    T = min(a.shape[0], b.shape[0])
    if T == 0:
        return np.nan
    a = a[:T]
    b = b[:T]
    # Use joint root-relative position (first 3 dims of 13-dim representation)
    a_pos = a[..., :3]
    b_pos = b[..., :3]
    a_flat = a_pos.reshape(T, -1)
    b_flat = b_pos.reshape(T, -1)
    a_c = a_flat - a_flat.mean(0, keepdims=True)
    b_c = b_flat - b_flat.mean(0, keepdims=True)
    a_norm = np.linalg.norm(a_c) + 1e-8
    b_norm = np.linalg.norm(b_c) + 1e-8
    a_n = a_c / a_norm
    b_n = b_c / b_norm
    M = a_n.T @ b_n
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt
    a_aligned = a_n @ R
    diff = a_aligned - b_n
    return float(np.sqrt((diff * diff).sum()))


# ---------- Latent distance (Frobenius) ----------

def _temporal_pool_to_fixed_shape(z):
    """Collapse the time/token axis so different source-clip lengths produce
    same-shape per-query latents. Strategy:

      Stage A latents (MoReFlow / ACE / AL-Flow): shape [T, D] → mean over T → [D]
      AnyTop encoder z: shape [J, F_lat, D] → mean over F_lat → [J, D]
        (J is fixed within a (skel_a) triple)
    """
    if z is None:
        return None
    if z.ndim == 2:
        return z.mean(axis=0)  # [T, D] -> [D]
    if z.ndim == 3:
        return z.mean(axis=1)  # [J, F_lat, D] -> [J, D]
    return z.flatten()  # fallback


def latent_distance(z_a, z_b):
    """Mean-pool to fixed shape, then Frobenius-normalised distance.

    Handles variable time/token length within a triple by collapsing time
    before comparison. The resulting metric measures average-direction
    similarity in latent space, which is the natural analogue to motion-space
    procrustes-aligned distance for SIF.
    """
    if z_a is None or z_b is None:
        return np.nan
    pa = _temporal_pool_to_fixed_shape(z_a)
    pb = _temporal_pool_to_fixed_shape(z_b)
    if pa.shape != pb.shape:
        return np.nan
    fa = pa.flatten()
    fb = pb.flatten()
    diff = fa - fb
    norm = np.linalg.norm(fa) + np.linalg.norm(fb) + 1e-8
    return float(np.linalg.norm(diff) / norm)


# ---------- Bootstrap CI ----------

def bootstrap_ci(values, n_boot=10000, seed=42):
    if not values:
        return [float('nan'), float('nan'), float('nan')]
    rng = np.random.RandomState(seed)
    arr = np.array(values)
    boots = [arr[rng.randint(0, len(arr), size=len(arr))].mean() for _ in range(n_boot)]
    return [float(np.percentile(boots, 2.5)), float(np.mean(arr)), float(np.percentile(boots, 97.5))]


# ---------- Main L-SIF computation ----------

def compute_lsif(latent_dir: Path, manifest, motion_dir: Path,
                 min_src_clips=2, max_src_clips=8, max_triples=None):
    queries = manifest['queries']
    by_triple = defaultdict(list)
    for q in queries:
        sa, sb, act = q.get('skel_a'), q.get('skel_b'), q.get('src_action')
        if sa and sb and act:
            by_triple[(sa, sb, act)].append(q)

    triples_used = []
    rhos = []

    for (sa, sb, act), qs in sorted(by_triple.items()):
        # Distinct source clips
        seen = {}
        for q in qs:
            sf = q.get('src_fname', '')
            if sf and sf not in seen:
                seen[sf] = q
        unique_qs = list(seen.values())
        if len(unique_qs) < min_src_clips:
            continue
        if len(unique_qs) > max_src_clips:
            unique_qs = unique_qs[:max_src_clips]

        # Load source motions
        src_motions = []
        for q in unique_qs:
            sp = motion_dir / q['src_fname']
            src_motions.append(np.load(sp).astype(np.float32) if sp.exists() else None)

        # Load target latents (from /tmp/{method}_latents/z_query_QID.npy)
        target_latents = []
        for q in unique_qs:
            qid = q.get('query_id')
            zp = latent_dir / f'z_query_{qid:04d}.npy'
            target_latents.append(np.load(zp) if zp.exists() else None)

        valid = [(s, l) for s, l in zip(src_motions, target_latents) if s is not None and l is not None]
        if len(valid) < min_src_clips:
            continue

        n = len(valid)
        D_src = np.full((n, n), np.nan)
        D_lat = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(i + 1, n):
                D_src[i, j] = D_src[j, i] = procrustes_3d_distance(valid[i][0], valid[j][0])
                D_lat[i, j] = D_lat[j, i] = latent_distance(valid[i][1], valid[j][1])

        mask = ~np.eye(n, dtype=bool) & ~np.isnan(D_src) & ~np.isnan(D_lat)
        if mask.sum() < 2:
            continue
        s_vec = D_src[mask]
        l_vec = D_lat[mask]
        if s_vec.std() < 1e-8 or l_vec.std() < 1e-8:
            rho = 0.0
        else:
            rho = float(np.corrcoef(s_vec, l_vec)[0, 1])
        if np.isnan(rho):
            continue
        triples_used.append({
            'skel_a': sa, 'skel_b': sb, 'action': act,
            'n': n, 'rho': rho,
            'src_dist_mean': float(s_vec.mean()),
            'lat_dist_mean': float(l_vec.mean()),
            'lat_over_src_diversity_ratio': float(l_vec.mean() / max(s_vec.mean(), 1e-8)),
        })
        rhos.append(rho)
        if max_triples and len(rhos) >= max_triples:
            break

    return triples_used, rhos


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', required=True)
    p.add_argument('--latent_dir', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--out_json', required=True)
    p.add_argument('--min_src_clips', type=int, default=2)
    p.add_argument('--max_src_clips', type=int, default=8)
    args = p.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    motion_dir = Path(DATASET_DIR) / 'motions'
    latent_dir = Path(args.latent_dir)

    print(f"[L-SIF] {args.method}: {latent_dir}")
    triples, rhos = compute_lsif(latent_dir, manifest, motion_dir,
                                  args.min_src_clips, args.max_src_clips)

    if rhos:
        ci = bootstrap_ci(rhos)
        agg = {
            'method': args.method,
            'n_triples': len(rhos),
            'lsif_mean': float(np.mean(rhos)),
            'lsif_ci_lo': ci[0],
            'lsif_ci_hi': ci[2],
            'lsif_median': float(np.median(rhos)),
            'lsif_min': float(np.min(rhos)),
            'lsif_max': float(np.max(rhos)),
        }
    else:
        agg = {'method': args.method, 'n_triples': 0}

    out = {'aggregate': agg, 'per_triple': triples}
    Path(args.out_json).write_text(json.dumps(out, indent=2, default=float))
    print(f"  saved {args.out_json}")
    print(f"  n_triples={agg.get('n_triples', 0)}, "
          f"L-SIF mean={agg.get('lsif_mean', float('nan')):.4f} "
          f"[{agg.get('lsif_ci_lo', float('nan')):.3f}, {agg.get('lsif_ci_hi', float('nan')):.3f}]")


if __name__ == '__main__':
    main()
