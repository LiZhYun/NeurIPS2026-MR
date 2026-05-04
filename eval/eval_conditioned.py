"""
Comprehensive evaluation for Motion-Conditioned AnyTop.

Covers:
  1. Self-reconstruction quality (encode X on S, decode on S)
  2. Cross-skeleton transfer (encode X on S, decode on T)
  3. Skeleton leakage probe (linear probe on z to predict skeleton ID)
  4. Failure analysis (identify high-loss samples, stratify by skeleton)

Usage:
    conda run -n anytop python -m eval.eval_conditioned \
        --model_path save/review_run_v1/model000050000.pt \
        --mode all
"""
import os
import json
import math
import random
import numpy as np
import torch
import torch.nn as nn
from argparse import ArgumentParser
from collections import defaultdict
from tqdm import tqdm

from utils.fixseed import fixseed
from utils import dist_util
from utils.model_util import create_conditioned_model_and_diffusion, load_model
from data_loaders.truebones.truebones_utils.get_opt import get_opt
from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np
from data_loaders.truebones.truebones_utils.plot_script import plot_general_skeleton_3d_motion
from data_loaders.tensors import truebones_batch_collate, create_padded_relation
from data_loaders.truebones.data.dataset import create_temporal_mask_for_window
from model.conditioners import T5Conditioner
from sample.generate_conditioned import (
    build_source_tensors, build_target_condition, encode_joints_names,
    load_args_from_checkpoint,
)
try:
    from eval.metrics.distances import avg_per_frame_dist
except ImportError:
    avg_per_frame_dist = None  # pytorch3d not available
from os.path import join as pjoin
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score


def load_model_and_cond(args):
    """Load trained model, diffusion, cond_dict, opt, and T5."""
    ckpt_args = load_args_from_checkpoint(args.model_path)
    class Namespace:
        def __init__(self, d):
            self.__dict__.update(d)
    ckpt = Namespace(ckpt_args)

    opt = get_opt(args.device)
    cond_dict = np.load(opt.cond_file, allow_pickle=True).item()

    model, diffusion = create_conditioned_model_and_diffusion(ckpt)
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model(model, state_dict)
    model.to(dist_util.dev())
    model.eval()

    t5_device = 'cpu' if args.cpu else 'cuda'
    t5_conditioner = T5Conditioner(
        name=ckpt.t5_name, finetune=False, word_dropout=0.0,
        normalize_text=False, device=t5_device)

    return model, diffusion, cond_dict, opt, ckpt, t5_conditioner


def get_all_motions_for_skeleton(skeleton_name, opt):
    """Return list of motion file paths for a given skeleton."""
    motion_dir = opt.motion_dir
    files = sorted(
        f for f in os.listdir(motion_dir)
        if f.startswith(f'{skeleton_name}_') and f.endswith('.npy'))
    return [pjoin(motion_dir, f) for f in files]


def load_and_normalize_motion(motion_path, cond_dict, skeleton_name, n_frames, max_joints):
    """Load, normalize, crop/pad a single motion to [1, J_max, 13, T]."""
    raw = np.load(motion_path)  # [T, J, 13]
    T_orig, J, _ = raw.shape

    mean = cond_dict[skeleton_name]['mean']
    std = cond_dict[skeleton_name]['std'] + 1e-6
    norm = (raw - mean[None, :]) / std[None, :]
    norm = np.nan_to_num(norm)

    if T_orig >= n_frames:
        norm = norm[:n_frames]
    else:
        pad = np.zeros((n_frames - T_orig, J, 13))
        norm = np.concatenate([norm, pad], axis=0)

    motion = np.zeros((n_frames, max_joints, 13))
    motion[:, :J, :] = norm
    offsets_raw = cond_dict[skeleton_name]['offsets']
    offsets = np.zeros((max_joints, 3))
    offsets[:J, :] = offsets_raw

    motion_t = torch.tensor(motion).permute(1, 2, 0).float().unsqueeze(0)  # [1, J_max, 13, T]
    offsets_t = torch.tensor(offsets).float().unsqueeze(0)                  # [1, J_max, 3]
    mask = torch.zeros(1, max_joints, dtype=torch.bool)
    mask[0, :J] = True

    return motion_t, offsets_t, mask, J, T_orig


# =============================================================================
# 1. Self-Reconstruction Evaluation
# =============================================================================

def eval_self_reconstruction(model, diffusion, cond_dict, opt, ckpt, t5_conditioner, args):
    """Evaluate self-reconstruction: encode X on S, decode on same S.

    Reports MSE, geodesic rotation error, stratified by skeleton.
    """
    print("\n" + "="*60)
    print("SELF-RECONSTRUCTION EVALUATION")
    print("="*60)

    n_frames = int(args.motion_length * opt.fps)
    device = dist_util.dev()
    all_skeletons = sorted(cond_dict.keys())
    if args.max_skeletons > 0:
        all_skeletons = all_skeletons[:args.max_skeletons]

    results = {}
    all_mse = []
    all_rot_err = []

    for skel_name in tqdm(all_skeletons, desc="Self-reconstruction"):
        motion_files = get_all_motions_for_skeleton(skel_name, opt)
        if not motion_files:
            continue

        skel_mse = []
        skel_rot_err = []
        n_joints = len(cond_dict[skel_name]['parents'])

        # Evaluate up to max_motions_per_skeleton
        for mf in motion_files[:args.max_motions_per_skeleton]:
            motion_t, offsets_t, mask, J, T_orig = load_and_normalize_motion(
                mf, cond_dict, skel_name, n_frames, opt.max_joints)
            motion_t = motion_t.to(device)
            offsets_t = offsets_t.to(device)
            mask = mask.to(device)

            with torch.no_grad():
                enc_out = model.encoder(motion_t, offsets_t, mask)
                z = enc_out[0] if isinstance(enc_out, tuple) else enc_out

            # Build target condition (same skeleton)
            _, model_kwargs = build_target_condition(
                skel_name, cond_dict, n_frames, ckpt.temporal_window,
                t5_conditioner, opt.max_joints, opt.feature_len)
            model_kwargs['y']['z'] = z

            with torch.no_grad():
                sample = diffusion.p_sample_loop(
                    model,
                    (1, opt.max_joints, opt.feature_len, n_frames),
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    skip_timesteps=0,
                    init_image=None,
                    progress=False,
                    dump_steps=None,
                    noise=None,
                    const_noise=False,
                )

            # Compare: both are normalized, so compare directly
            source = motion_t[0, :n_joints].cpu()   # [J, 13, T]
            recon = sample[0, :n_joints].cpu()       # [J, 13, T]

            # MSE on position channels (0:3)
            pos_mse = ((source[:, :3, :] - recon[:, :3, :]) ** 2).mean().item()
            # MSE on rotation channels (3:9)
            rot_mse = ((source[:, 3:9, :] - recon[:, 3:9, :]) ** 2).mean().item()
            total_mse = ((source - recon) ** 2).mean().item()

            skel_mse.append(total_mse)
            skel_rot_err.append(rot_mse)

        if skel_mse:
            results[skel_name] = {
                'mse_mean': np.mean(skel_mse),
                'mse_std': np.std(skel_mse),
                'rot_mse_mean': np.mean(skel_rot_err),
                'n_joints': n_joints,
                'n_motions': len(skel_mse),
            }
            all_mse.extend(skel_mse)
            all_rot_err.extend(skel_rot_err)
            print(f"  {skel_name}: MSE={np.mean(skel_mse):.4f} (n={len(skel_mse)}, J={n_joints})")

    # Summary
    print(f"\nOVERALL: MSE={np.mean(all_mse):.4f} +/- {np.std(all_mse):.4f}")
    print(f"         Rot MSE={np.mean(all_rot_err):.4f} +/- {np.std(all_rot_err):.4f}")

    # Stratify by joint count
    low_j = [v['mse_mean'] for v in results.values() if v['n_joints'] <= 30]
    mid_j = [v['mse_mean'] for v in results.values() if 30 < v['n_joints'] <= 60]
    high_j = [v['mse_mean'] for v in results.values() if v['n_joints'] > 60]
    if low_j:
        print(f"  J<=30: MSE={np.mean(low_j):.4f} (n={len(low_j)} skeletons)")
    if mid_j:
        print(f"  30<J<=60: MSE={np.mean(mid_j):.4f} (n={len(mid_j)} skeletons)")
    if high_j:
        print(f"  J>60: MSE={np.mean(high_j):.4f} (n={len(high_j)} skeletons)")

    return results


# =============================================================================
# 2. Cross-Skeleton Transfer Evaluation
# =============================================================================

TRANSFER_PAIRS = [
    # Within-category (quadruped → quadruped)
    ('Horse', 'Cat'),
    ('Cat', 'Coyote'),
    ('Elephant', 'Bear'),
    ('Lion', 'Fox'),
    # Cross-category
    ('Horse', 'Raptor'),       # quadruped → biped
    ('Cat', 'Spider'),         # quadruped → milliped
    ('Dragon', 'Eagle'),       # flying → flying
    ('Coyote', 'Flamingo'),    # quadruped → biped
    ('Bear', 'Trex'),          # quadruped → biped
]


def eval_cross_skeleton_transfer(model, diffusion, cond_dict, opt, ckpt, t5_conditioner, args):
    """Evaluate cross-skeleton transfer with qualitative outputs and quantitative metrics."""
    print("\n" + "="*60)
    print("CROSS-SKELETON TRANSFER EVALUATION")
    print("="*60)

    n_frames = int(args.motion_length * opt.fps)
    device = dist_util.dev()
    out_dir = pjoin(os.path.dirname(args.model_path), 'cross_skeleton_eval')
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    for src_skel, tgt_skel in TRANSFER_PAIRS:
        if src_skel not in cond_dict or tgt_skel not in cond_dict:
            print(f"  Skipping {src_skel} -> {tgt_skel}: skeleton not in dataset")
            continue

        print(f"\n  {src_skel} -> {tgt_skel}")

        # Encode source
        motion_t, offsets_t, mask = build_source_tensors(
            src_skel, cond_dict, opt, n_frames, ckpt.temporal_window)
        motion_t = motion_t.to(device)
        offsets_t = offsets_t.to(device)
        mask = mask.to(device)

        with torch.no_grad():
            enc_out = model.encoder(motion_t, offsets_t, mask)
            z = enc_out[0] if isinstance(enc_out, tuple) else enc_out

        # Also generate unconditional on target (for comparison)
        _, model_kwargs_cond = build_target_condition(
            tgt_skel, cond_dict, n_frames, ckpt.temporal_window,
            t5_conditioner, opt.max_joints, opt.feature_len)
        model_kwargs_cond['y']['z'] = z

        _, model_kwargs_uncond = build_target_condition(
            tgt_skel, cond_dict, n_frames, ckpt.temporal_window,
            t5_conditioner, opt.max_joints, opt.feature_len)
        # uncond: no z → model uses null_z

        shape = (1, opt.max_joints, opt.feature_len, n_frames)

        with torch.no_grad():
            sample_cond = diffusion.p_sample_loop(
                model, shape, clip_denoised=False,
                model_kwargs=model_kwargs_cond, progress=False)
            sample_uncond = diffusion.p_sample_loop(
                model, shape, clip_denoised=False,
                model_kwargs=model_kwargs_uncond, progress=False)

        tgt = cond_dict[tgt_skel]
        n_joints = len(tgt['parents'])

        # Denormalize
        motion_cond = sample_cond[0, :n_joints].cpu().permute(2, 0, 1).numpy()  # [T, J, 13]
        motion_cond = motion_cond * tgt['std'][None, :] + tgt['mean'][None, :]
        motion_uncond = sample_uncond[0, :n_joints].cpu().permute(2, 0, 1).numpy()
        motion_uncond = motion_uncond * tgt['std'][None, :] + tgt['mean'][None, :]

        # Denormalize source for comparison
        src = cond_dict[src_skel]
        n_src_joints = len(src['parents'])
        src_motion = motion_t[0, :n_src_joints].cpu().permute(2, 0, 1).numpy()
        src_motion = src_motion * src['std'][None, :] + src['mean'][None, :]

        # Compute metrics
        # 1. Conditioned vs unconditioned difference (should be nonzero if z has effect)
        cond_uncond_diff = np.mean((motion_cond - motion_uncond) ** 2)

        # 2. Velocity profile correlation (tempo preservation)
        src_vel = np.linalg.norm(np.diff(src_motion[:, :, :3], axis=0), axis=-1).mean(axis=1)  # [T-1]
        tgt_vel = np.linalg.norm(np.diff(motion_cond[:, :, :3], axis=0), axis=-1).mean(axis=1)
        min_len = min(len(src_vel), len(tgt_vel))
        if min_len > 5:
            vel_corr = np.corrcoef(src_vel[:min_len], tgt_vel[:min_len])[0, 1]
        else:
            vel_corr = float('nan')

        # 3. Energy profile correlation
        src_energy = np.sum(src_motion[:, :, 9:12] ** 2, axis=(1, 2))  # velocity channels
        tgt_energy = np.sum(motion_cond[:, :, 9:12] ** 2, axis=(1, 2))
        min_len = min(len(src_energy), len(tgt_energy))
        if min_len > 5:
            energy_corr = np.corrcoef(src_energy[:min_len], tgt_energy[:min_len])[0, 1]
        else:
            energy_corr = float('nan')

        results[f"{src_skel}->{tgt_skel}"] = {
            'cond_uncond_diff': cond_uncond_diff,
            'velocity_correlation': vel_corr,
            'energy_correlation': energy_corr,
        }
        print(f"    Cond vs Uncond MSE: {cond_uncond_diff:.4f}")
        print(f"    Velocity correlation: {vel_corr:.4f}")
        print(f"    Energy correlation: {energy_corr:.4f}")

        # Save videos
        try:
            global_pos_cond = recover_from_bvh_ric_np(motion_cond)
            plot_general_skeleton_3d_motion(
                pjoin(out_dir, f'{src_skel}_to_{tgt_skel}_cond.mp4'),
                tgt['parents'], global_pos_cond,
                title=f'{src_skel} -> {tgt_skel} (conditioned)', fps=opt.fps)

            global_pos_uncond = recover_from_bvh_ric_np(motion_uncond)
            plot_general_skeleton_3d_motion(
                pjoin(out_dir, f'{src_skel}_to_{tgt_skel}_uncond.mp4'),
                tgt['parents'], global_pos_uncond,
                title=f'{src_skel} -> {tgt_skel} (unconditioned)', fps=opt.fps)

            global_pos_src = recover_from_bvh_ric_np(src_motion)
            plot_general_skeleton_3d_motion(
                pjoin(out_dir, f'{src_skel}_source.mp4'),
                src['parents'], global_pos_src,
                title=f'{src_skel} (source)', fps=opt.fps)
        except Exception as e:
            print(f"    Video export failed: {e}")

        # Save NPY
        np.save(pjoin(out_dir, f'{src_skel}_to_{tgt_skel}_cond.npy'), motion_cond)
        np.save(pjoin(out_dir, f'{src_skel}_to_{tgt_skel}_uncond.npy'), motion_uncond)

    # Summary
    print("\n--- Cross-Skeleton Transfer Summary ---")
    for pair, metrics in results.items():
        print(f"  {pair}: vel_corr={metrics['velocity_correlation']:.3f}, "
              f"energy_corr={metrics['energy_correlation']:.3f}, "
              f"cond_uncond_diff={metrics['cond_uncond_diff']:.4f}")

    return results


# =============================================================================
# 3. Skeleton Leakage Probe
# =============================================================================

def eval_skeleton_leakage_probe(model, cond_dict, opt, ckpt, args):
    """Train a linear probe on z to predict skeleton identity.

    If z is truly skeleton-agnostic, the probe should perform at chance level.
    """
    print("\n" + "="*60)
    print("SKELETON LEAKAGE PROBE")
    print("="*60)

    n_frames = int(args.motion_length * opt.fps)
    device = dist_util.dev()

    all_z = []
    all_labels = []
    all_n_joints = []

    all_skeletons = sorted(cond_dict.keys())

    for skel_name in tqdm(all_skeletons, desc="Encoding motions for probe"):
        motion_files = get_all_motions_for_skeleton(skel_name, opt)
        if not motion_files:
            continue

        for mf in motion_files[:args.max_motions_per_skeleton]:
            motion_t, offsets_t, mask, J, _ = load_and_normalize_motion(
                mf, cond_dict, skel_name, n_frames, opt.max_joints)
            motion_t = motion_t.to(device)
            offsets_t = offsets_t.to(device)
            mask = mask.to(device)

            with torch.no_grad():
                enc_out = model.encoder(motion_t, offsets_t, mask)
                z = enc_out[0] if isinstance(enc_out, tuple) else enc_out  # [1, N', K, D']

            # Flatten z to a single vector per sample
            z_flat = z.view(1, -1).cpu().numpy()  # [1, N'*K*D']
            all_z.append(z_flat[0])
            all_labels.append(skel_name)
            all_n_joints.append(J)

    X = np.array(all_z)
    y_skel = np.array(all_labels)
    y_joints = np.array(all_n_joints)

    print(f"\nProbe dataset: {X.shape[0]} samples, {len(set(y_skel))} skeleton types")
    print(f"z dimensionality: {X.shape[1]}")

    # Only include skeletons with >= 2 samples for cross-validation
    skel_counts = defaultdict(int)
    for s in y_skel:
        skel_counts[s] += 1
    valid_mask = np.array([skel_counts[s] >= 2 for s in y_skel])
    X_valid = X[valid_mask]
    y_skel_valid = y_skel[valid_mask]
    y_joints_valid = y_joints[valid_mask]

    print(f"After filtering (>=2 samples): {X_valid.shape[0]} samples, "
          f"{len(set(y_skel_valid))} skeleton types")

    if len(X_valid) < 10:
        print("Too few samples for reliable probing. Skipping.")
        return {}

    # Probe 1: Skeleton ID prediction
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_skel_valid)
    n_classes = len(le.classes_)
    chance = 1.0 / n_classes

    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    n_folds = min(5, min(np.bincount(y_encoded)))
    if n_folds >= 2:
        scores = cross_val_score(clf, X_valid, y_encoded, cv=n_folds, scoring='accuracy')
        skel_acc = scores.mean()
    else:
        clf.fit(X_valid, y_encoded)
        skel_acc = clf.score(X_valid, y_encoded)

    print(f"\n  Skeleton ID probe accuracy: {skel_acc:.3f} (chance = {chance:.3f})")
    print(f"  Ratio above chance: {skel_acc / chance:.2f}x")

    # Probe 2: Joint count prediction (regression)
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score as cvs_reg
    reg = Ridge(alpha=1.0)
    if len(X_valid) >= 10:
        r2_scores = cvs_reg(reg, X_valid, y_joints_valid, cv=min(5, len(X_valid)//2),
                           scoring='r2')
        joint_r2 = r2_scores.mean()
        print(f"  Joint count regression R2: {joint_r2:.3f}")
    else:
        joint_r2 = float('nan')

    # Probe 3: Coarse morphology (category) prediction
    from data_loaders.truebones.truebones_utils.param_utils import (
        QUADROPEDS, BIPEDS, MILLIPEDS, FLYING, SNAKES, FISH)
    category_map = {}
    for s in QUADROPEDS:
        category_map[s] = 'quadruped'
    for s in BIPEDS:
        category_map[s] = 'biped'
    for s in MILLIPEDS:
        category_map[s] = 'milliped'
    for s in FLYING:
        category_map[s] = 'flying'
    for s in SNAKES:
        category_map[s] = 'snake'
    for s in FISH:
        category_map[s] = 'fish'

    y_cat = np.array([category_map.get(s, 'unknown') for s in y_skel_valid])
    valid_cat = y_cat != 'unknown'
    if valid_cat.sum() >= 10:
        le_cat = LabelEncoder()
        y_cat_enc = le_cat.fit_transform(y_cat[valid_cat])
        n_cat = len(le_cat.classes_)
        chance_cat = 1.0 / n_cat

        clf_cat = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        n_folds_cat = min(5, min(np.bincount(y_cat_enc)))
        if n_folds_cat >= 2:
            scores_cat = cross_val_score(clf_cat, X_valid[valid_cat], y_cat_enc,
                                         cv=n_folds_cat, scoring='accuracy')
            cat_acc = scores_cat.mean()
        else:
            clf_cat.fit(X_valid[valid_cat], y_cat_enc)
            cat_acc = clf_cat.score(X_valid[valid_cat], y_cat_enc)

        print(f"  Category probe accuracy: {cat_acc:.3f} (chance = {chance_cat:.3f}, "
              f"categories={list(le_cat.classes_)})")
    else:
        cat_acc = float('nan')

    results = {
        'skeleton_id_accuracy': skel_acc,
        'skeleton_id_chance': chance,
        'joint_count_r2': joint_r2,
        'category_accuracy': cat_acc if not np.isnan(cat_acc) else None,
        'n_samples': len(X_valid),
        'n_skeleton_types': n_classes,
    }

    return results


# =============================================================================
# 4. Failure Analysis
# =============================================================================

def eval_failure_analysis(model, diffusion, cond_dict, opt, ckpt, t5_conditioner, args):
    """Identify which skeletons/motions have highest reconstruction error."""
    print("\n" + "="*60)
    print("FAILURE ANALYSIS")
    print("="*60)

    n_frames = int(args.motion_length * opt.fps)
    device = dist_util.dev()

    # Compute per-sample diffusion loss (no generation needed — just forward pass)
    all_losses = []
    all_skeletons_list = sorted(cond_dict.keys())

    for skel_name in tqdm(all_skeletons_list, desc="Computing losses"):
        motion_files = get_all_motions_for_skeleton(skel_name, opt)
        if not motion_files:
            continue

        n_joints = len(cond_dict[skel_name]['parents'])

        for mf in motion_files[:args.max_motions_per_skeleton]:
            motion_t, offsets_t, mask, J, T_orig = load_and_normalize_motion(
                mf, cond_dict, skel_name, n_frames, opt.max_joints)
            motion_t = motion_t.to(device)
            offsets_t = offsets_t.to(device)
            mask = mask.to(device)

            with torch.no_grad():
                enc_out = model.encoder(motion_t, offsets_t, mask)
                z = enc_out[0] if isinstance(enc_out, tuple) else enc_out

            # Build condition and move to device
            _, model_kwargs = build_target_condition(
                skel_name, cond_dict, n_frames, ckpt.temporal_window,
                t5_conditioner, opt.max_joints, opt.feature_len)
            model_kwargs['y'] = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in model_kwargs['y'].items()}
            model_kwargs['y']['z'] = z

            # Compute training loss at multiple timesteps
            losses_at_t = []
            for _ in range(5):  # average over 5 random timesteps
                t = torch.randint(0, 100, (1,), device=device)
                with torch.no_grad():
                    loss_dict = diffusion.training_losses(
                        model, motion_t, t, model_kwargs=model_kwargs)
                losses_at_t.append(loss_dict['loss'].item())

            avg_loss = np.mean(losses_at_t)
            all_losses.append({
                'skeleton': skel_name,
                'motion_file': os.path.basename(mf),
                'loss': avg_loss,
                'n_joints': n_joints,
                'T_orig': T_orig,
            })

    # Sort by loss
    all_losses.sort(key=lambda x: x['loss'], reverse=True)

    print(f"\n--- Top 10 Highest Loss Samples ---")
    for i, entry in enumerate(all_losses[:10]):
        print(f"  {i+1}. {entry['skeleton']} / {entry['motion_file']}: "
              f"loss={entry['loss']:.4f} (J={entry['n_joints']}, T={entry['T_orig']})")

    print(f"\n--- Top 10 Lowest Loss Samples ---")
    for i, entry in enumerate(all_losses[-10:]):
        print(f"  {i+1}. {entry['skeleton']} / {entry['motion_file']}: "
              f"loss={entry['loss']:.4f} (J={entry['n_joints']}, T={entry['T_orig']})")

    # Stratify by skeleton
    skel_losses = defaultdict(list)
    for entry in all_losses:
        skel_losses[entry['skeleton']].append(entry['loss'])

    print(f"\n--- Per-Skeleton Average Loss (top 10 worst) ---")
    skel_avg = [(s, np.mean(v), len(v)) for s, v in skel_losses.items()]
    skel_avg.sort(key=lambda x: x[1], reverse=True)
    for s, avg, n in skel_avg[:10]:
        n_j = cond_dict[s]['parents'].shape[0] if hasattr(cond_dict[s]['parents'], 'shape') else len(cond_dict[s]['parents'])
        print(f"  {s}: loss={avg:.4f} (n={n}, J={n_j})")

    # Correlation between joint count and loss
    joints = [e['n_joints'] for e in all_losses]
    losses = [e['loss'] for e in all_losses]
    corr = np.corrcoef(joints, losses)[0, 1]
    print(f"\nCorrelation(n_joints, loss): {corr:.3f}")

    return {
        'all_losses': all_losses,
        'per_skeleton': {s: {'avg_loss': avg, 'n_motions': n} for s, avg, n in skel_avg},
        'joint_count_loss_correlation': corr,
    }


# =============================================================================
# 5. CFG Effect Analysis
# =============================================================================

def eval_cfg_effect(model, diffusion, cond_dict, opt, ckpt, t5_conditioner, args):
    """Evaluate whether z conditioning has any effect by comparing conditioned vs unconditioned."""
    print("\n" + "="*60)
    print("CFG EFFECT ANALYSIS")
    print("="*60)

    n_frames = int(args.motion_length * opt.fps)
    device = dist_util.dev()

    test_skeletons = ['Horse', 'Cat', 'Dog', 'Spider', 'Raptor', 'Dragon', 'Eagle', 'Bear']
    test_skeletons = [s for s in test_skeletons if s in cond_dict]

    results = {}
    for skel_name in test_skeletons:
        motion_t, offsets_t, mask = build_source_tensors(
            skel_name, cond_dict, opt, n_frames, ckpt.temporal_window)
        motion_t = motion_t.to(device)
        offsets_t = offsets_t.to(device)
        mask = mask.to(device)

        with torch.no_grad():
            enc_out = model.encoder(motion_t, offsets_t, mask)
            z = enc_out[0] if isinstance(enc_out, tuple) else enc_out

        # Forward pass at t=50 (mid-diffusion)
        t = torch.tensor([50], device=device)
        noise = torch.randn_like(motion_t)

        from diffusion.gaussian_diffusion import _extract_into_tensor
        sqrt_alpha = _extract_into_tensor(diffusion.sqrt_alphas_cumprod, t, motion_t.shape)
        sqrt_one_minus = _extract_into_tensor(diffusion.sqrt_one_minus_alphas_cumprod, t, motion_t.shape)
        x_t = sqrt_alpha * motion_t + sqrt_one_minus * noise

        _, model_kwargs = build_target_condition(
            skel_name, cond_dict, n_frames, ckpt.temporal_window,
            t5_conditioner, opt.max_joints, opt.feature_len)

        with torch.no_grad():
            # Conditioned prediction
            model_kwargs['y']['z'] = z
            pred_cond = model(x_t, t, y=model_kwargs['y'])

            # Unconditioned prediction (null z)
            import copy
            uncond_kwargs = copy.deepcopy(model_kwargs)
            uncond_kwargs['y']['z'] = model.null_z.expand(1, -1, -1, -1)
            pred_uncond = model(x_t, t, y=uncond_kwargs['y'])

        diff = ((pred_cond - pred_uncond) ** 2).mean().item()
        results[skel_name] = diff
        print(f"  {skel_name}: cond-uncond MSE = {diff:.6f}")

    avg_diff = np.mean(list(results.values()))
    print(f"\n  Average cond-uncond difference: {avg_diff:.6f}")
    if avg_diff < 1e-4:
        print("  WARNING: z appears to have negligible effect on decoder output!")
    else:
        print("  z conditioning is active (decoder responds differently to z vs null).")

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--mode', default='all',
                        choices=['all', 'self_recon', 'cross_skeleton', 'leakage', 'failure', 'cfg'],
                        help='Which evaluation to run')
    parser.add_argument('--motion_length', default=5.0, type=float)
    parser.add_argument('--max_skeletons', default=0, type=int,
                        help='Max skeletons for self-recon (0=all)')
    parser.add_argument('--max_motions_per_skeleton', default=3, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--cpu', action='store_true',
                        help='Run entirely on CPU (when GPU is busy with training)')
    args = parser.parse_args()

    fixseed(args.seed)
    if args.cpu:
        dist_util.setup_dist(-1)  # negative device → CPU in dist_util.dev()
    else:
        dist_util.setup_dist(args.device)

    print("Loading model...")
    model, diffusion, cond_dict, opt, ckpt, t5_conditioner = load_model_and_cond(args)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")
    print(f"Dataset: {len(cond_dict)} skeleton types")

    all_results = {}

    if args.mode in ('all', 'cfg'):
        all_results['cfg_effect'] = eval_cfg_effect(
            model, diffusion, cond_dict, opt, ckpt, t5_conditioner, args)

    if args.mode in ('all', 'leakage'):
        all_results['leakage_probe'] = eval_skeleton_leakage_probe(
            model, cond_dict, opt, ckpt, args)

    if args.mode in ('all', 'failure'):
        all_results['failure_analysis'] = eval_failure_analysis(
            model, diffusion, cond_dict, opt, ckpt, t5_conditioner, args)

    if args.mode in ('all', 'self_recon'):
        all_results['self_reconstruction'] = eval_self_reconstruction(
            model, diffusion, cond_dict, opt, ckpt, t5_conditioner, args)

    if args.mode in ('all', 'cross_skeleton'):
        all_results['cross_skeleton'] = eval_cross_skeleton_transfer(
            model, diffusion, cond_dict, opt, ckpt, t5_conditioner, args)

    # Save results
    out_path = pjoin(os.path.dirname(args.model_path), 'eval_results.json')
    # Convert numpy types to native Python for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(out_path, 'w') as f:
        json.dump(convert(all_results), f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
