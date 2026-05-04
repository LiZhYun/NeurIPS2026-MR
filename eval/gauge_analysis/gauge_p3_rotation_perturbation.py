"""P3 (decoder-nondegeneracy) rotation perturbation, generic over Stage-A-decoded methods.

Tests the decoder-nondegeneracy assumption empirically: take a saved latent z,
apply a random O(d) rotation R, decode both R·z and z through the per-target-skeleton
Stage A decoder, measure the fractional motion-space divergence vs a matched-magnitude
noise-direction control.

Theorem prediction (§5.23): under marginal-matching latent training, the latent gauge
ambiguity transfers to OBSERVABLE decoded outputs only if the decoder is non-degenerate
(decode(R·z) ≠ decode(z) on positive measure). This script measures the gap empirically
for any method whose latents follow the per-target-skeleton Stage-A interface (ACE,
MoReFlow, AL-Flow).

Usage:
    python /tmp/gauge_p3_rotation_perturbation.py \
        --latent_dir /tmp/ace_latents/no_ladv_seed42 \
        --method_name ACE_no_L_adv \
        --n_rotations 20 --out_json /tmp/gauge_p3_ace_no_ladv.json

Output schema:
    {method_name, n_queries_processed, per_query: [{rel_div_rotation, rel_div_noise, ratio, ...}]}
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

from data_loaders.truebones.truebones_utils.param_utils import OBJECT_SUBSETS_DICT
from model.moreflow.stage_a_registry import StageARegistry


def random_orthogonal(d, rng):
    """Random matrix uniformly drawn from O(d) via QR of standard normal."""
    A = rng.randn(d, d).astype(np.float32)
    Q, R = np.linalg.qr(A)
    # Make R diagonal positive; sign matters for exact O(d) sampling
    Q = Q * np.sign(np.diag(R))[None, :]
    return Q  # [d, d]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_dir', required=True,
                        help='Directory with z_query_*.npy + meta.json')
    parser.add_argument('--method_name', required=True)
    parser.add_argument('--n_rotations', type=int, default=20)
    parser.add_argument('--n_noise', type=int, default=20)
    parser.add_argument('--out_json', required=True)
    parser.add_argument('--rng_seed', type=int, default=42)
    parser.add_argument('--limit_queries', type=int, default=None,
                        help='Process only first N queries (debug)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[P3] Device: {device}")
    print(f"[P3] Loading Stage A registry...")
    all_skels = list(set(OBJECT_SUBSETS_DICT['train_v3']) | set(OBJECT_SUBSETS_DICT['test_v3']))
    registry = StageARegistry(all_skels, device=device)

    latent_dir = Path(args.latent_dir)
    meta = json.loads((latent_dir / 'meta.json').read_text())
    if args.limit_queries:
        meta = meta[:args.limit_queries]
    print(f"[P3] Processing {len(meta)} queries from {latent_dir}")

    rng = np.random.RandomState(args.rng_seed)
    per_query = []
    t0 = time.time()
    with torch.no_grad():
        for entry in meta:
            qid = entry['query_id']
            tgt_skel = entry['tgt_skel']
            try:
                z = np.load(latent_dir / f'z_query_{qid:04d}.npy')  # [T, d]
                if z.ndim != 2:
                    print(f"  [{qid}] skip: z shape {z.shape} != 2D")
                    continue
                T, d = z.shape
                z_t = torch.from_numpy(z).to(device).unsqueeze(0)  # [1, T, d]

                # Decode baseline
                base_norm = registry.decode_tokens(tgt_skel, z_t)
                base_phys = registry.unnormalize(tgt_skel, base_norm)  # [1, T, J, 13]
                base_np = base_phys.squeeze(0).cpu().numpy()
                base_norm_sq = float(np.sum(base_np ** 2))

                # P3 (rotation perturbation) — n_rotations samples
                rot_divs = []
                for _ in range(args.n_rotations):
                    R = random_orthogonal(d, rng)
                    z_rot_np = z @ R  # [T, d]
                    z_rot = torch.from_numpy(z_rot_np.astype(np.float32)).to(device).unsqueeze(0)
                    decoded_norm = registry.decode_tokens(tgt_skel, z_rot)
                    decoded_phys = registry.unnormalize(tgt_skel, decoded_norm)
                    diff_sq = float(torch.sum((decoded_phys - base_phys) ** 2).item())
                    rot_divs.append(diff_sq / max(base_norm_sq, 1e-12))

                # Noise-floor control — random-direction perturbation of matched magnitude
                # Magnitude target: ||(R-I)z||_F / ||z||_F (relative perturbation in latent)
                # We compute it once per query from the first rotation as the reference scale.
                R_ref = random_orthogonal(d, rng)
                pert_ref = z @ R_ref - z
                pert_mag = float(np.linalg.norm(pert_ref) / max(np.linalg.norm(z), 1e-12))
                noise_divs = []
                for _ in range(args.n_noise):
                    eps = rng.randn(T, d).astype(np.float32)
                    eps = eps * (pert_mag * np.linalg.norm(z) / max(np.linalg.norm(eps), 1e-12))
                    z_noise = z + eps
                    z_noise_t = torch.from_numpy(z_noise.astype(np.float32)).to(device).unsqueeze(0)
                    decoded_norm = registry.decode_tokens(tgt_skel, z_noise_t)
                    decoded_phys = registry.unnormalize(tgt_skel, decoded_norm)
                    diff_sq = float(torch.sum((decoded_phys - base_phys) ** 2).item())
                    noise_divs.append(diff_sq / max(base_norm_sq, 1e-12))

                rot_mean = float(np.mean(rot_divs))
                noise_mean = float(np.mean(noise_divs))
                ratio = rot_mean / max(noise_mean, 1e-12)

                per_query.append({
                    'query_id': qid, 'tgt_skel': tgt_skel,
                    'd': d, 'T': T,
                    'rel_div_rotation_mean': rot_mean,
                    'rel_div_rotation_std': float(np.std(rot_divs)),
                    'rel_div_noise_mean': noise_mean,
                    'rel_div_noise_std': float(np.std(noise_divs)),
                    'rotation_over_noise_ratio': ratio,
                    'pert_mag_relative': pert_mag,
                })
            except Exception as e:
                print(f"  [{qid}] FAILED: {type(e).__name__}: {e}")
            if (len(per_query) % 25 == 0) and per_query:
                print(f"  [{len(per_query)}/{len(meta)}] done ({time.time() - t0:.0f}s)")

    # Aggregate
    if per_query:
        rot = np.array([r['rel_div_rotation_mean'] for r in per_query])
        noise = np.array([r['rel_div_noise_mean'] for r in per_query])
        ratios = np.array([r['rotation_over_noise_ratio'] for r in per_query])
        agg = {
            'n_queries': len(per_query),
            'rel_div_rotation_mean': float(rot.mean()),
            'rel_div_rotation_std': float(rot.std()),
            'rel_div_noise_mean': float(noise.mean()),
            'rel_div_noise_std': float(noise.std()),
            'ratio_mean': float(ratios.mean()),
            'ratio_std': float(ratios.std()),
            'ratio_median': float(np.median(ratios)),
            'ratio_min': float(ratios.min()),
            'ratio_max': float(ratios.max()),
            'verdict': 'NON-DEGENERATE' if ratios.mean() > 1.5 else 'WEAK / NEAR NOISE FLOOR',
        }
    else:
        agg = {'n_queries': 0}

    out = {
        'method_name': args.method_name,
        'latent_dir': args.latent_dir,
        'n_rotations_per_query': args.n_rotations,
        'n_noise_per_query': args.n_noise,
        'aggregate': agg,
        'per_query': per_query,
    }
    Path(args.out_json).write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[P3] saved {args.out_json}")
    print(f"=== AGGREGATE for {args.method_name} ===")
    for k, v in agg.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == '__main__':
    main()
