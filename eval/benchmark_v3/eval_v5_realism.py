"""V5 realism / efficiency evaluator (Tier 1 metrics + Tier 2 FID).

Per-method, per-fold, per-query metrics:

  Tier 1 (all methods):
    - foot_skate     : per-frame foot velocity when contact=1 (lower is better)
    - bone_validity  : mean abs deviation of bone lengths from rest-pose (lower is better)
    - contact_F1     : F1 of predicted contact pattern vs nearest positive_cluster ref (higher is better)
    - jerk           : third derivative magnitude of positions (lower = smoother)
    - freq_align     : PSD cosine sim of motion vs source per-joint, averaged (higher is better)
    - inference_fps  : queries per second from method's metrics.json (higher is better)

  Tier 2 (generative baselines only — retrieval methods get FID ≈ 0 trivially):
    - fid_per_skel   : Fréchet distance between method's outputs (per skel) and real target-skel clip distribution

Aggregation: per-fold mean across queries; both folds averaged.

Usage:
  python -m eval.benchmark_v3.eval_v5_realism \
      --method_dir save/m3/m3_rerank_v1 --method_name M3_PhaseA_v1
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
from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np

DATA_ROOT = Path(DATASET_DIR)
MOTION_DIR = DATA_ROOT / 'motions'
COND_PATH = DATA_ROOT / 'cond.npy'
QUERIES_V5_DIR = PROJECT_ROOT / 'eval/benchmark_v3/queries_v5'
CONTACT_GROUPS_PATH = PROJECT_ROOT / 'eval/quotient_assets/contact_groups.json'


# ---------------- Helpers ----------------

def get_positions(motion: np.ndarray, n_joints: int):
    """Return positions [T, J, 3] from motion (handles both [T,J,3] and [T,J,13] inputs)."""
    if motion.shape[1] > n_joints:
        motion = motion[:, :n_joints]
    if motion.ndim == 3 and motion.shape[-1] == 3:
        return motion.astype(np.float32)
    if motion.ndim == 3 and motion.shape[-1] == 13:
        return recover_from_bvh_ric_np(motion.astype(np.float32))
    raise ValueError(f'Unexpected motion shape: {motion.shape}')


def get_contact_channel(motion: np.ndarray, n_joints: int):
    """Per-joint binary contact signal [T, J] from motion[..., :, 12]. Returns None if positions-only."""
    if motion.shape[1] > n_joints:
        motion = motion[:, :n_joints]
    if motion.ndim == 3 and motion.shape[-1] == 13:
        return (motion[..., :, 12] > 0.5).astype(np.float32)
    return None


def foot_skate(positions: np.ndarray, contact: np.ndarray, foot_idxs: list, fps: int = 30) -> float:
    """Per-foot, per-frame: ||vel(foot)|| when contact[foot] = 1.
    Average over (T-1) frames and feet. positions: [T, J, 3]."""
    if not foot_idxs:
        return float('nan')
    T, J, _ = positions.shape
    vel = (positions[1:] - positions[:-1]) * fps  # [T-1, J, 3]
    speed = np.linalg.norm(vel, axis=-1)            # [T-1, J]
    foot_idxs_valid = [i for i in foot_idxs if 0 <= i < J]
    if not foot_idxs_valid:
        return float('nan')
    foot_speed = speed[:, foot_idxs_valid]          # [T-1, F]
    if contact is not None:
        # Use motion's own contact channel
        c = contact[1:, foot_idxs_valid]            # [T-1, F]
    else:
        # Heuristic: contact when foot is below 25th percentile of its height range
        h = positions[:, foot_idxs_valid, 1]        # [T, F] — y-axis = up
        c = np.zeros_like(foot_speed)
        for fi in range(len(foot_idxs_valid)):
            thr = np.percentile(h[:, fi], 25)
            c[:, fi] = (h[1:, fi] < thr).astype(np.float32)
    if c.sum() < 1: return 0.0
    skate = (foot_speed * c).sum() / max(c.sum(), 1)
    return float(skate)


def bone_validity(positions: np.ndarray, parents: list, rest_offsets: np.ndarray) -> float:
    """Mean abs deviation of bone lengths from rest-pose offsets, normalized by rest length.
    positions: [T, J, 3], parents: list, rest_offsets: [J, 3]."""
    T, J, _ = positions.shape
    rest_lengths = np.linalg.norm(rest_offsets, axis=-1)  # [J]
    devs = []
    for j in range(J):
        p = parents[j]
        if not (0 <= p < J) or p == j: continue
        if rest_lengths[j] < 1e-3: continue
        bone_vec = positions[:, j] - positions[:, p]
        bone_len = np.linalg.norm(bone_vec, axis=-1)  # [T]
        rel_dev = np.abs(bone_len - rest_lengths[j]) / rest_lengths[j]
        devs.append(rel_dev.mean())
    if not devs: return float('nan')
    return float(np.mean(devs))


def jerk(positions: np.ndarray, fps: int = 30) -> float:
    """Mean magnitude of 3rd derivative of positions (jerk)."""
    if positions.shape[0] < 4: return float('nan')
    accel = (positions[2:] - 2 * positions[1:-1] + positions[:-2]) * (fps ** 2)
    jrk = (accel[1:] - accel[:-1]) * fps  # [T-3, J, 3]
    return float(np.linalg.norm(jrk, axis=-1).mean())


def contact_f1_vs_reference(pred_contact: np.ndarray, ref_contact: np.ndarray,
                            contact_groups: dict, joint_count: int) -> float:
    """Per-contact-group F1 averaged.
    pred_contact: [T_p, J] from method, ref_contact: [T_r, J] from reference.
    Resample reference to same length.
    """
    if pred_contact is None or ref_contact is None: return float('nan')
    T_p, J = pred_contact.shape
    if pred_contact.shape[1] != ref_contact.shape[1]:
        # Use min joint count (shouldn't happen if same skel)
        J = min(pred_contact.shape[1], ref_contact.shape[1])
        pred_contact = pred_contact[:, :J]
        ref_contact = ref_contact[:, :J]
    # Resample ref to T_p frames via linear interpolation
    if ref_contact.shape[0] != T_p:
        from scipy.interpolate import interp1d
        xs_in = np.linspace(0, 1, ref_contact.shape[0])
        xs_out = np.linspace(0, 1, T_p)
        ref_resamp = np.zeros((T_p, J))
        for j in range(J):
            f = interp1d(xs_in, ref_contact[:, j], kind='nearest', assume_sorted=True)
            ref_resamp[:, j] = f(xs_out)
        ref_contact = ref_resamp
    pred_contact = (pred_contact > 0.5).astype(np.int8)
    ref_contact = (ref_contact > 0.5).astype(np.int8)
    f1s = []
    for group_name, joint_idxs in contact_groups.items():
        valid = [j for j in joint_idxs if 0 <= j < J]
        if not valid: continue
        # Collapse to per-frame per-group: any joint in group in contact → group active
        p = pred_contact[:, valid].max(axis=1)
        r = ref_contact[:, valid].max(axis=1)
        tp = ((p == 1) & (r == 1)).sum()
        fp = ((p == 1) & (r == 0)).sum()
        fn = ((p == 0) & (r == 1)).sum()
        if tp + fp + fn == 0: continue
        f1 = (2 * tp) / (2 * tp + fp + fn + 1e-9)
        f1s.append(f1)
    if not f1s: return float('nan')
    return float(np.mean(f1s))


def freq_align_psd(src_positions: np.ndarray, pred_positions: np.ndarray, fps: int = 30) -> float:
    """PSD cosine sim of source vs predicted motion, averaged across joints.
    Different joint counts: take min(J_src, J_pred). Different lengths: pad/truncate to min."""
    T_s, J_s, _ = src_positions.shape
    T_p, J_p, _ = pred_positions.shape
    T = min(T_s, T_p); J = min(J_s, J_p)
    s = src_positions[:T, :J]
    p = pred_positions[:T, :J]
    # Per-joint PSD (FFT power spectrum of magnitude)
    s_mag = np.linalg.norm(s, axis=-1)  # [T, J]
    p_mag = np.linalg.norm(p, axis=-1)
    # Detrend and FFT
    s_mag = s_mag - s_mag.mean(axis=0, keepdims=True)
    p_mag = p_mag - p_mag.mean(axis=0, keepdims=True)
    psd_s = np.abs(np.fft.rfft(s_mag, axis=0)) ** 2  # [F, J]
    psd_p = np.abs(np.fft.rfft(p_mag, axis=0)) ** 2
    cos_sims = []
    for j in range(J):
        a, b = psd_s[:, j], psd_p[:, j]
        na = np.linalg.norm(a); nb = np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9: continue
        cos_sims.append(float(a @ b / (na * nb)))
    if not cos_sims: return float('nan')
    return float(np.mean(cos_sims))


# ---------------- FID (Tier 2) ----------------

def fid_score(features_real: np.ndarray, features_gen: np.ndarray) -> float:
    """Fréchet distance between two distributions (mean+cov of features).
    features_real, features_gen: [N, D] each.
    """
    from scipy.linalg import sqrtm
    if features_real.shape[0] < 2 or features_gen.shape[0] < 2:
        return float('nan')
    mu_r = features_real.mean(axis=0)
    mu_g = features_gen.mean(axis=0)
    sig_r = np.cov(features_real, rowvar=False)
    sig_g = np.cov(features_gen, rowvar=False)
    diff = mu_r - mu_g
    covmean = sqrtm(sig_r @ sig_g)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sig_r + sig_g - 2 * covmean))


def kinematic_features_per_clip(positions: np.ndarray) -> np.ndarray:
    """Compact per-clip kinematic feature: COM trajectory stats + velocity stats + joint velocity stats.
    Per-clip 30-d feature vector (skeleton-agnostic).
    """
    T, J, _ = positions.shape
    com = positions.mean(axis=1)  # [T, 3]
    com_vel = np.diff(com, axis=0) if T > 1 else np.zeros((1, 3))
    joint_vel = np.diff(positions, axis=0) if T > 1 else np.zeros((1, J, 3))
    feat = np.concatenate([
        com.mean(axis=0), com.std(axis=0),                          # 6
        com_vel.mean(axis=0), com_vel.std(axis=0),                  # 6
        np.array([joint_vel.mean(), joint_vel.std()]),              # 2
        np.array([np.abs(joint_vel).mean()]),                       # 1
        np.percentile(np.linalg.norm(joint_vel, axis=-1), [25, 50, 75]),  # 3
        np.array([np.linalg.norm(positions, axis=-1).mean(), np.linalg.norm(positions, axis=-1).std()]),  # 2
        np.array([positions[:, :, 1].mean(), positions[:, :, 1].std(), positions[:, :, 1].max(), positions[:, :, 1].min()]),  # 4 (height)
        np.array([T / 60.0, J / 50.0]),                             # 2 (length, joints)
    ])
    # Pad to 30
    if feat.shape[0] < 30:
        feat = np.concatenate([feat, np.zeros(30 - feat.shape[0])])
    return feat[:30].astype(np.float32)


# ---------------- Main eval ----------------

def evaluate_method_realism(method_dir: Path, fold: int, method_name: str,
                             cond_dict: dict, contact_groups: dict,
                             real_features_per_skel: dict = None,
                             max_queries: int = 10000):
    """Compute per-query Tier 1 metrics + (optionally) per-skel FID.
    Returns dict {metric: value, per_query: list}."""
    manifest = json.load(open(QUERIES_V5_DIR / f'fold_{fold}/manifest.json'))
    per_query = []
    n_skipped = 0

    # For per-skel FID accumulation
    gen_features_per_skel = defaultdict(list)

    for q in manifest['queries'][:max_queries]:
        qid = q['query_id']
        skel_b = q['skel_b']
        pred_path = method_dir / f'query_{qid:04d}.npy'
        if not pred_path.exists():
            n_skipped += 1; continue
        try:
            pred = np.load(pred_path).astype(np.float32)
            cond_b = cond_dict[skel_b]
            n_joints_b = len(cond_b['parents'])
            pred_positions = get_positions(pred, n_joints_b)
            pred_contact = get_contact_channel(pred, n_joints_b)

            # Source positions for freq align
            src_motion = np.load(MOTION_DIR / q['src_fname']).astype(np.float32)
            cond_a = cond_dict[q['skel_a']]
            src_positions = get_positions(src_motion, len(cond_a['parents']))

            # Reference for contact_F1: nearest positive_cluster (first one)
            ref_contact = None
            pos_clips = q.get('positives_cluster', [])
            if pos_clips:
                ref_motion = np.load(MOTION_DIR / pos_clips[0]['fname']).astype(np.float32)
                ref_contact = get_contact_channel(ref_motion, n_joints_b)

            # Tier 1 metrics
            cg_b = contact_groups.get(skel_b, {})
            foot_idxs = []
            for k, v in cg_b.items():
                if not k.startswith('_'):
                    foot_idxs.extend(v)
            foot_idxs = sorted(set(foot_idxs))

            r_foot_skate = foot_skate(pred_positions, pred_contact, foot_idxs)
            r_bone_valid = bone_validity(pred_positions, cond_b['parents'], cond_b['offsets'])
            r_jerk = jerk(pred_positions)
            r_freq_align = freq_align_psd(src_positions, pred_positions)
            r_contact_f1 = contact_f1_vs_reference(pred_contact, ref_contact, cg_b, n_joints_b)

            per_query.append({
                'query_id': qid,
                'skel_b': skel_b,
                'split': q['split'],
                'cluster': q['cluster'],
                'foot_skate': r_foot_skate,
                'bone_validity': r_bone_valid,
                'jerk': r_jerk,
                'freq_align': r_freq_align,
                'contact_f1': r_contact_f1,
            })

            # Tier 2: collect per-skel features
            gen_features_per_skel[skel_b].append(kinematic_features_per_clip(pred_positions))

        except Exception as e:
            n_skipped += 1
            print(f'  q{qid}: FAIL {e}')

    # Aggregate
    def safe_mean(vals):
        v = [x for x in vals if x is not None and not np.isnan(x)]
        if not v: return float('nan')
        return float(np.mean(v))

    out = {
        'method': method_name, 'fold': fold,
        'n_queries': len(per_query), 'n_skipped': n_skipped,
        'per_query': per_query,
    }
    for split in ('all', 'test_test'):
        prefix = '' if split == 'all' else 'test_test_'
        rows = [r for r in per_query if split == 'all' or r['split'] == split]
        for m in ('foot_skate', 'bone_validity', 'jerk', 'freq_align', 'contact_f1'):
            out[f'{prefix}{m}'] = safe_mean([r[m] for r in rows])

    # Tier 2 FID
    if real_features_per_skel is not None:
        fids = []
        per_skel_fid = {}
        for skel_b, gen_list in gen_features_per_skel.items():
            if skel_b not in real_features_per_skel: continue
            real_arr = real_features_per_skel[skel_b]
            gen_arr = np.stack(gen_list)
            if gen_arr.shape[0] < 2 or real_arr.shape[0] < 2: continue
            f = fid_score(real_arr, gen_arr)
            per_skel_fid[skel_b] = f
            fids.append(f)
        out['fid_per_skel'] = per_skel_fid
        out['fid_mean_skels'] = float(np.mean(fids)) if fids else float('nan')
        out['fid_n_skels'] = len(fids)

    return out


def build_real_features_per_skel(cond_dict: dict, max_per_skel: int = 50):
    """Pre-compute kinematic features for all real clips per skeleton (for FID's 'real' distribution)."""
    real = defaultdict(list)
    motion_files = list(MOTION_DIR.glob('*.npy'))
    print(f"Loading {len(motion_files)} real motion clips for FID baseline...")
    for f in motion_files:
        skel = f.name.split('___')[0] if '___' in f.name else f.name.split('_')[0]
        if skel not in cond_dict: continue
        if len(real[skel]) >= max_per_skel: continue
        try:
            m = np.load(f).astype(np.float32)
            n_j = len(cond_dict[skel]['parents'])
            pos = get_positions(m, n_j)
            real[skel].append(kinematic_features_per_clip(pos))
        except Exception:
            continue
    real_arrays = {s: np.stack(feats) for s, feats in real.items() if len(feats) >= 2}
    print(f"Built features for {len(real_arrays)} skels (avg {np.mean([a.shape[0] for a in real_arrays.values()]):.1f} clips/skel)")
    return real_arrays


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method_dir', required=True, help='Method dir with fold_NN/query_NNNN.npy')
    parser.add_argument('--method_name', required=True)
    parser.add_argument('--folds', nargs='+', type=int, default=[42, 43])
    parser.add_argument('--out_path', default=None)
    parser.add_argument('--include_fid', action='store_true', help='Compute Tier 2 FID (slower)')
    args = parser.parse_args()

    cond_dict = np.load(COND_PATH, allow_pickle=True).item()
    contact_groups = json.load(open(CONTACT_GROUPS_PATH))

    real_features = None
    if args.include_fid:
        real_features = build_real_features_per_skel(cond_dict)

    method_root = Path(args.method_dir)
    if not method_root.is_absolute():
        method_root = PROJECT_ROOT / method_root

    all_results = {}
    for fold in args.folds:
        # Try fold_NN, then v3_fold_NN
        fdir = method_root / f'fold_{fold}'
        if not fdir.exists():
            fdir = method_root / f'v3_fold_{fold}'
        if not fdir.exists():
            print(f"WARN: no dir for fold {fold} under {method_root}"); continue
        print(f"Eval fold {fold}: {fdir}")
        r = evaluate_method_realism(fdir, fold, args.method_name, cond_dict,
                                     contact_groups, real_features)
        all_results[f'fold_{fold}'] = r
        print(f"  fold {fold}: foot_skate={r['foot_skate']:.4f} bone_validity={r['bone_validity']:.4f}")
        print(f"             jerk={r['jerk']:.2f} freq_align={r['freq_align']:.4f}")
        print(f"             contact_f1={r['contact_f1']:.4f}")
        print(f"  test_test:  foot_skate={r['test_test_foot_skate']:.4f} "
              f"bone_validity={r['test_test_bone_validity']:.4f}")
        if 'fid_mean_skels' in r:
            print(f"  fid_mean_skels (n_skels={r['fid_n_skels']}): {r['fid_mean_skels']:.4f}")

    out_path = Path(args.out_path) if args.out_path else method_root / 'realism_eval.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
