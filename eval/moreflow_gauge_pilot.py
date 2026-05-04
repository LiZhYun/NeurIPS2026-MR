"""Falsification pilot: do per-char Stage A codebooks share a gauge?

For each (skel_A, skel_B) pair of morphologically-similar quadrupeds:
  1. Encode skel_A's motion → z tokens (continuous codebook vectors)
  2. Decode through skel_A's decoder → x_AA (proper reconstruction baseline)
  3. Decode through skel_B's decoder → x_AB (cross-decoded — the test)
  4. Decode random target tokens through skel_B → x_RB (random baseline)

If gauges are aligned: x_AB resembles a coherent motion (smooth, structured),
maybe even semantically related to source.

If gauges are independent: x_AB looks like x_RB (jittery garbage).

Numeric proxies (no rendering required):
- Frame-to-frame velocity magnitude (smaller = smoother)
- Velocity autocorrelation at lag 1 (higher = more coherent)
- Per-joint position std over time (smaller = less jittery within joint)
- Cross-vs-self decode similarity in invariant Q-features

Usage:
  python -m eval.moreflow_gauge_pilot
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.moreflow.vqvae import MoReFlowVQVAE

DATA_ROOT = PROJECT_ROOT / 'dataset/truebones/zoo/truebones_processed'
MOTION_DIR = DATA_ROOT / 'motions'
COND_PATH = DATA_ROOT / 'cond.npy'
CKPT_ROOT = PROJECT_ROOT / 'save/moreflow_vqvae'

# Morphologically-similar quadruped pairs (J close, both 4-legged)
PAIRS = [
    ('Horse', 'Bear'),       # J=79 vs 76
    ('Camel', 'Jaguar'),     # J=50 vs 45
    ('BrownBear', 'Lion'),   # J=38 vs 31
]
N_CLIPS_PER_PAIR = 4
WINDOW = 32


def load_tokenizer(skel_name, device):
    ckpt_path = CKPT_ROOT / skel_name / 'ckpt_best.pt'
    if not ckpt_path.exists():
        ckpt_path = CKPT_ROOT / skel_name / 'ckpt_final.pt'
    sd = torch.load(ckpt_path, map_location=device)
    a = sd['args']
    model = MoReFlowVQVAE(
        input_dim=sd['feat_dim'],
        hidden=a['hidden'],
        codebook_size=sd['codebook_size_effective'],
        codebook_dim=a['codebook_dim'],
        n_resblocks=a['n_resblocks'],
        n_downsample=a['n_downsample'],
        activation=a['activation'],
        dead_code_threshold=a.get('dead_code_threshold', 5),
    ).to(device).eval()
    model.load_state_dict(sd['model_state_dict'])
    mean = torch.from_numpy(np.asarray(sd['mean'], dtype=np.float32)).to(device)
    std = torch.from_numpy(np.asarray(sd['std'], dtype=np.float32) + 1e-6).to(device)
    return model, mean, std, sd['n_joints']


def list_clips(skel_name):
    return sorted([f for f in MOTION_DIR.iterdir()
                   if f.name.startswith(skel_name + '___')
                   and f.suffix == '.npy'])


def velocity_stats(motion_13d):
    """motion_13d: [T, J, 13]. Use first 3 dims (positions)."""
    pos = motion_13d[..., :3]  # [T, J, 3]
    vel = pos[1:] - pos[:-1]   # [T-1, J, 3]
    vmag = np.linalg.norm(vel, axis=-1)  # [T-1, J]
    return {
        'vmag_mean': float(vmag.mean()),
        'vmag_std': float(vmag.std()),
        # autocorr at lag 1: how similar consecutive velocities are
        'vel_autocorr_lag1': float(_autocorr(vel.reshape(vel.shape[0], -1), lag=1)),
        'pos_std_per_joint_mean': float(np.linalg.norm(pos.std(axis=0), axis=-1).mean()),
    }


def _autocorr(x, lag=1):
    """x: [T, D]. Returns mean per-dim autocorr at given lag."""
    if x.shape[0] <= lag + 2:
        return float('nan')
    a = x[:-lag]
    b = x[lag:]
    a_centered = a - a.mean(axis=0, keepdims=True)
    b_centered = b - b.mean(axis=0, keepdims=True)
    num = (a_centered * b_centered).sum(axis=0)
    den = np.sqrt((a_centered**2).sum(axis=0) * (b_centered**2).sum(axis=0)) + 1e-9
    return float((num / den).mean())


def encode_decode(model_src, mean_src, std_src, model_tgt, mean_tgt, std_tgt,
                  motion_13d, device):
    """Encode through src tokenizer, decode through tgt tokenizer.

    motion_13d: [T_src, J_src, 13] np
    Returns: [T_src, J_tgt, 13] np
    """
    T = motion_13d.shape[0]
    # Crop to multiple of downsample_factor
    T_crop = (T // model_src.downsample_factor) * model_src.downsample_factor
    motion = torch.from_numpy(motion_13d[:T_crop]).float().to(device)
    motion_norm = (motion - mean_src.unsqueeze(0)) / std_src.unsqueeze(0)
    motion_norm = torch.nan_to_num(motion_norm)
    motion_flat = motion_norm.reshape(T_crop, -1).unsqueeze(0)  # [1, T, J*13]

    with torch.no_grad():
        z_q, indices, _ = model_src.encode(motion_flat)
        # z_q: [1, T/4, codebook_dim] — continuous quantized codebook vectors
        recon = model_tgt.decode(z_q)  # [1, T, J_tgt*13]
    recon = recon.squeeze(0).reshape(T_crop, model_tgt.input_dim // 13, 13)
    # Un-normalize via tgt skel's stats
    recon_un = recon * std_tgt.unsqueeze(0) + mean_tgt.unsqueeze(0)
    return recon_un.cpu().numpy()


def random_decode(model_tgt, mean_tgt, std_tgt, T, device):
    """Decode random codebook indices through tgt tokenizer."""
    K_eff = model_tgt.vq.codebook_size
    n_tok = T // model_tgt.downsample_factor
    rng_idx = torch.randint(0, K_eff, (1, n_tok), device=device)
    with torch.no_grad():
        z_q = model_tgt.vq.get_codes_from_indices(rng_idx)
        recon = model_tgt.decode(z_q)
    T_out = recon.shape[1]
    recon = recon.squeeze(0).reshape(T_out, model_tgt.input_dim // 13, 13)
    recon_un = recon * std_tgt.unsqueeze(0) + mean_tgt.unsqueeze(0)
    return recon_un.cpu().numpy()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Pairs: {PAIRS}")
    print(f"N clips per pair: {N_CLIPS_PER_PAIR}\n")

    results = {}
    rng = np.random.RandomState(42)

    for skel_a, skel_b in PAIRS:
        print(f"\n{'='*60}")
        print(f"Pair: {skel_a} (encoder) → {skel_b} (decoder)")
        print('='*60)
        model_a, mean_a, std_a, J_a = load_tokenizer(skel_a, device)
        model_b, mean_b, std_b, J_b = load_tokenizer(skel_b, device)
        print(f"  J_a={J_a}, J_b={J_b}")
        print(f"  K_a={model_a.vq.codebook_size}, K_b={model_b.vq.codebook_size}")

        clips = list_clips(skel_a)
        rng.shuffle(clips)
        clips = clips[:N_CLIPS_PER_PAIR]

        per_pair_stats = []
        for clip_path in clips:
            motion = np.load(clip_path).astype(np.float32)
            if motion.shape[0] < WINDOW * 2:
                continue
            print(f"\n  Clip: {clip_path.name} ({motion.shape[0]} frames)")

            # 1. Self-decode (baseline — should be ~original)
            x_AA = encode_decode(model_a, mean_a, std_a, model_a, mean_a, std_a, motion, device)
            # 2. Cross-decode (the test — should be coherent if gauge aligned)
            x_AB = encode_decode(model_a, mean_a, std_a, model_b, mean_b, std_b, motion, device)
            # 3. Random-decode through B (chance baseline)
            x_RB = random_decode(model_b, mean_b, std_b, motion.shape[0], device)

            stats_orig = velocity_stats(motion)
            stats_AA = velocity_stats(x_AA)
            stats_AB = velocity_stats(x_AB)
            stats_RB = velocity_stats(x_RB)

            print(f"    {'metric':<20} {'orig_a':>10} {'self_a':>10} {'cross_b':>10} {'rand_b':>10}")
            for k in ['vmag_mean', 'vmag_std', 'vel_autocorr_lag1', 'pos_std_per_joint_mean']:
                print(f"    {k:<20} {stats_orig[k]:>10.4f} {stats_AA[k]:>10.4f} {stats_AB[k]:>10.4f} {stats_RB[k]:>10.4f}")

            per_pair_stats.append({
                'clip': clip_path.name,
                'orig': stats_orig, 'self_a': stats_AA, 'cross_b': stats_AB, 'rand_b': stats_RB,
            })

        results[f"{skel_a}__{skel_b}"] = per_pair_stats
        # free GPU
        del model_a, model_b
        torch.cuda.empty_cache()

    out_path = PROJECT_ROOT / 'save/moreflow_gauge_pilot_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n\nSaved {out_path}")

    # Aggregate verdict
    print(f"\n{'='*60}\nAGGREGATE — cross-decode vs random-decode (lower = better)\n{'='*60}")
    print(f"{'pair':<25} {'cross_vmag_mean':>17} {'rand_vmag_mean':>17} {'ratio':>8}")
    for pair_name, trials in results.items():
        if not trials:
            continue
        cross_vmag = np.mean([t['cross_b']['vmag_mean'] for t in trials])
        rand_vmag = np.mean([t['rand_b']['vmag_mean'] for t in trials])
        ratio = cross_vmag / max(rand_vmag, 1e-6)
        verdict = "GAUGE-LIKELY-ALIGNED" if ratio < 0.5 else "AMBIGUOUS" if ratio < 1.0 else "GAUGE-INDEPENDENT"
        print(f"{pair_name:<25} {cross_vmag:>17.4f} {rand_vmag:>17.4f} {ratio:>8.3f}  {verdict}")

    print(f"\n{'='*60}\nAGGREGATE — cross-decode autocorrelation (higher = more coherent)\n{'='*60}")
    print(f"{'pair':<25} {'orig_acorr':>12} {'cross_acorr':>12} {'rand_acorr':>12}")
    for pair_name, trials in results.items():
        if not trials:
            continue
        orig_a = np.mean([t['orig']['vel_autocorr_lag1'] for t in trials])
        cross_a = np.mean([t['cross_b']['vel_autocorr_lag1'] for t in trials])
        rand_a = np.mean([t['rand_b']['vel_autocorr_lag1'] for t in trials])
        print(f"{pair_name:<25} {orig_a:>12.4f} {cross_a:>12.4f} {rand_a:>12.4f}")


if __name__ == '__main__':
    main()
