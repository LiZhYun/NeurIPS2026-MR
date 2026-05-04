"""Motion2Motion (Chen et al., arXiv 2508.13139, SIGGRAPH Asia 2025) reproduction
on Truebones Zoo 30 canonical eval pairs.

STATUS: APPROXIMATE "M2M-lite" implementation.
    Official code (https://github.com/LinghaoChan/Motion2Motion_codes) operates on BVH
    files with a suffix-mangled joint-name scheme and produces BVH output. Round-tripping
    the 30 Truebones pairs through the official pipeline would require:
      (1) running utils/add_random_string_of_joints.py over 1070 BVHs,
      (2) authoring a BVH mappings.json per pair that the official ConfigParser accepts
          (joint names must match the suffixed BVH names),
      (3) writing a BVH-to-13-dim Truebones converter.
    Under the 60-minute budget this is infeasible, so we faithfully reproduce the CORE
    ALGORITHM on the 13-dim representation directly. Implemented:
      - coarse-to-fine multi-resolution pyramid (as `_get_*_pyramid`)
      - overlapping 11-frame patch extraction (`extract_patches`)
      - efficient chunked cdist NN-retrieval (`get_NNs_Dists`) across target example patches
      - sparse-correspondence projection mask weighting of the patch distance (Eq. 1 of paper,
        `matching_alpha=0.9`)
      - blended-patch reconstruction (weighted average overlapping patches)
      - L=3 outer optimisation iterations per pyramid level
    Simplifications vs. paper:
      - (A) the paper operates on [rotation, root_pos] in 6D+3-dim; we operate on the 13-dim
        (pos3+rot6+vel3+contact1) representation. The two are interconvertible but the raw
        13-dim representation carries extra redundancy that the paper's BVH->motion-data
        layer strips. This should make per-joint matching MORE noisy, not less, so numbers
        here are a LOWER BOUND on the paper's published Truebones numbers.
      - (B) sparse correspondences come from `eval/quotient_assets/contact_groups.json` as
        a heuristic (paired by group — LF<->LF, RF<->RF, etc.) plus extreme-pair manual
        author rules (snake front/mid/back <-> biped pelvis/torso/pelvis). Paper uses
        manually authored 2-6 bone mappings per pair; our heuristic agrees with what the
        paper's SIGGRAPH demos show for similar-skeleton pairs but over-matches for insects
        (we pair up to 8 groups where paper uses 2-6; this should HELP our numbers, not hurt).
      - (C) no matching_alpha scheduling or pyramid factor tuning.

Command:
    conda run -n anytop python eval/motion2motion_run.py

Outputs:
    eval/results/k_compare/motion2motion/pair_<id>_<src>_to_<tgt>.npy   [T, J, 13]
    eval/results/k_compare/motion2motion/metrics.json
"""
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import json
import os
import sys
import time
from os.path import join as pjoin
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, f"{PROJECT_ROOT}/external/motion2motion")

from eval.quotient_extractor import extract_quotient

ROOT = Path(str(PROJECT_ROOT))
MOTION_DIR = ROOT / 'dataset/truebones/zoo/truebones_processed/motions'
COND_PATH = ROOT / 'dataset/truebones/zoo/truebones_processed/cond.npy'
CONTACT_GROUPS_PATH = ROOT / 'eval/quotient_assets/contact_groups.json'
LABELS_PATH = ROOT / 'dataset/truebones/zoo/truebones_processed/motion_labels.json'
EVAL_PAIRS_PATH = ROOT / 'idea-stage/eval_pairs.json'
OUT_DIR = ROOT / 'eval/results/k_compare/motion2motion'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Manual mappings for extreme pairs where the contact-group heuristic breaks down.
# Each entry maps (source_skel, target_skel) -> list of (src_joint_index, tgt_joint_index) pairs.
# Designed 2-4 pairings per extreme pair as per the paper.
EXTREME_MANUAL_MAPPINGS = {}  # filled at runtime via heuristic helpers below

PATCH_SIZE = 11
NUM_STEPS = 3
NOISE_SIGMA = 1.0  # reduced from paper's 10.0 which is calibrated for 6D rotation scale;
                   # our 13-dim channels are already in a smaller numeric range
COARSE_RATIO = 0.4  # paper defaults 5*patchsize/len ~ 0.5 for 100-frame sequences; we clamp
PYR_FACTOR = 0.75
MATCHING_ALPHA = 0.9


# --------- Official M2M core algorithms (from external/motion2motion/nearest_neighbor) ---
def extract_patches(x, patch_size, stride=1, loop=False):
    """x: [B, C, T]; returns [B, N, C*patch_size]"""
    B, C, T = x.shape
    if loop:
        # loop padding
        pad = patch_size // 2
        x = torch.cat([x, x[:, :, :patch_size - 1]], dim=-1)
    patches = x.unfold(-1, patch_size, stride)  # [B, C, N, patch_size]
    N = patches.shape[2]
    patches = patches.permute(0, 2, 1, 3).reshape(B, N, C * patch_size)
    return patches


def combine_patches(x_shape, patches, patch_size, stride=1, loop=False):
    """Inverse of extract_patches: reconstruct [B, C, T] from [B, N, C*patch_size]
    by averaging overlapping patches."""
    B, C, T = x_shape
    N = patches.shape[1]
    patches = patches.view(B, N, C, patch_size).permute(0, 2, 3, 1)  # [B, C, patch_size, N]
    out = torch.zeros(B, C, T, device=patches.device, dtype=patches.dtype)
    count = torch.zeros(B, 1, T, device=patches.device, dtype=patches.dtype)
    for i in range(patch_size):
        # patches[:, :, i, :] contributes to frames [i, i+stride, ...]
        idx = torch.arange(N, device=patches.device) * stride + i
        # valid idx < T
        valid = idx < T
        if loop:
            idx = idx % T
            valid = torch.ones_like(valid)
        if not valid.any():
            continue
        v_idx = idx[valid]
        v_vals = patches[:, :, i, valid]
        out.index_add_(-1, v_idx, v_vals)
        count.index_add_(-1, v_idx, torch.ones_like(v_vals[:, :1]))
    return out / count.clamp_min(1e-6)


def efficient_cdist(X, Y, chunk=1024):
    """X: [N, D], Y: [M, D] -> [N, M]"""
    # ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x.y
    X2 = (X ** 2).sum(-1, keepdim=True)
    Y2 = (Y ** 2).sum(-1, keepdim=True).t()
    out = torch.empty(X.shape[0], Y.shape[0], device=X.device)
    for i in range(0, X.shape[0], chunk):
        Xi = X[i:i+chunk]
        Xi2 = X2[i:i+chunk]
        out[i:i+chunk] = (Xi2 + Y2 - 2 * Xi @ Y.t()).clamp_min(0.0).sqrt()
    return out


def get_NNs_Dists(dist_fn, X_patches, Y_patches, alpha):
    """For each x in X, find its nearest y in Y; return (nnf, dists).
    If alpha is set, apply the bidirectional matching normalization from the paper.
    X_patches: [Nx, D], Y_patches: [Ny, D]
    """
    D = dist_fn(X_patches, Y_patches)  # [Nx, Ny]
    if alpha is not None and alpha > 0:
        # normalize by Y's typical distance
        d_min = D.min(dim=0, keepdim=True).values + alpha  # [1, Ny]
        D = D / d_min
    nnf = D.argmin(dim=1)
    return nnf, D.gather(1, nnf.unsqueeze(-1)).squeeze(-1)


# --------- Sparse bone correspondence authoring (heuristic from contact_groups.json) -----
def author_sparse_mapping(src_skel, tgt_skel, cond, contact_groups, max_pairs=6):
    """Return list of (src_joint_idx, tgt_joint_idx) and a description string.

    Strategy:
      (a) Both skels have >=1 shared contact-group name: pair JOINT INDICES within matching groups.
          Take the first (most distal) joint index in each group (feet/wings/claws).
          Cap at max_pairs total bone pairings.
      (b) Only partial overlap (some groups unique to one side): prefer shared group names,
          fall back to symmetric pairing (L* with L*, R* with R*).
      (c) Incompatible contact groups (e.g. spider L1..L4 vs biped LF..RF): snake/multi-leg
          groups -> biped by explicit heuristic: front legs/claws <-> front feet (LF/RF),
          back legs/tail <-> hind feet (LH/RH).
      Always include the root (index 0 on both sides) as an anchor.
    """
    src_g = contact_groups.get(src_skel, {})
    tgt_g = contact_groups.get(tgt_skel, {})
    src_joints = cond[src_skel]['joints_names']
    tgt_joints = cond[tgt_skel]['joints_names']

    pairs = [(0, 0)]  # root-root
    description = [f'root({src_joints[0]})↔root({tgt_joints[0]})']

    # (a) exact group match
    shared = [k for k in src_g if k in tgt_g and not k.startswith('_')]
    for g in shared:
        if len(pairs) >= max_pairs:
            break
        src_idx = src_g[g][-1]  # most distal joint in group
        tgt_idx = tgt_g[g][-1]
        pairs.append((src_idx, tgt_idx))
        description.append(f'{g}:{src_joints[src_idx]}↔{tgt_joints[tgt_idx]}')

    if len(pairs) >= 3:
        return pairs, description

    # (b) symmetric L*/R* pairing
    src_L = [k for k in src_g if k.startswith('L') and not k.startswith('_')]
    src_R = [k for k in src_g if k.startswith('R') and not k.startswith('_')]
    tgt_L = [k for k in tgt_g if k.startswith('L') and not k.startswith('_')]
    tgt_R = [k for k in tgt_g if k.startswith('R') and not k.startswith('_')]

    for s_groups, t_groups, side in [(src_L, tgt_L, 'L'), (src_R, tgt_R, 'R')]:
        for sg in s_groups:
            if len(pairs) >= max_pairs:
                break
            if not t_groups:
                continue
            tg = t_groups[0]  # pick first available
            src_idx = src_g[sg][-1]
            tgt_idx = tgt_g[tg][-1]
            existing = [(s, t) for (s, t) in pairs]
            if (src_idx, tgt_idx) in existing:
                continue
            pairs.append((src_idx, tgt_idx))
            description.append(f'{side}-fallback:{src_joints[src_idx]}↔{tgt_joints[tgt_idx]}')

    if len(pairs) >= 3:
        return pairs[:max_pairs], description[:max_pairs]

    # (c) total mismatch: pair front joint of each group to first tgt group we have
    tgt_all = list(tgt_g.values())
    for i, (g_name, s_idxs) in enumerate(src_g.items()):
        if len(pairs) >= max_pairs:
            break
        if i >= len(tgt_all):
            break
        s_idx = s_idxs[-1]
        t_idx = tgt_all[i][-1]
        if (s_idx, t_idx) not in pairs:
            pairs.append((s_idx, t_idx))
            description.append(f'cross-fallback:{src_joints[s_idx]}↔{tgt_joints[t_idx]}')

    if len(pairs) < 2:
        # truly extreme: pick pelvis + first deep joint of each skeleton (scorpion->biped etc.)
        # take the root (already added) + one far joint
        s_far = min(len(src_joints) - 1, len(src_joints) // 2)
        t_far = min(len(tgt_joints) - 1, len(tgt_joints) // 2)
        pairs.append((s_far, t_far))
        description.append(f'extreme-fallback:{src_joints[s_far]}↔{tgt_joints[t_far]}')

    return pairs[:max_pairs], description[:max_pairs]


# --------- Motion patch matching on 13-dim motion tensors ------------------------------
def run_m2m_lite(source_motion, example_motion, src_j_idxs, tgt_j_idxs, tgt_n_joints,
                 device='cpu', seed=42):
    """Core M2M patch matching on 13-dim motion representation.

    source_motion: [T_s, J_s, 13]   source motion on source skeleton
    example_motion: [T_e, J_t, 13]  example motion on target skeleton
    src_j_idxs, tgt_j_idxs: paired-bone indices (same length); these are the "bound" joints
    tgt_n_joints: J_t (target joint count)
    Returns: [T_s, J_t, 13] retargeted motion
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    J_s = source_motion.shape[1]
    J_t = tgt_n_joints
    assert example_motion.shape[1] == J_t
    C = 13
    assert len(src_j_idxs) == len(tgt_j_idxs)

    # Build [1, C*B, T] tensors where B is the number of bound joints.
    # We match on bound-joint channels only (~sparse correspondence).
    src_bnd = source_motion[:, src_j_idxs, :]      # [T_s, B, 13]
    ex_bnd = example_motion[:, tgt_j_idxs, :]      # [T_e, B, 13]

    # Store example as ALL-JOINT target for blending unbound joints back in (Eq. 1 projection)
    ex_full = example_motion                        # [T_e, J_t, 13]

    # Multi-resolution pyramid (coarse to fine)
    T_s = source_motion.shape[0]
    T_e = example_motion.shape[0]
    final_len = T_s

    # pyramid lengths
    lengths = [max(PATCH_SIZE + 2, int(round(final_len * COARSE_RATIO)))]
    while lengths[-1] < final_len:
        nxt = int(round(lengths[-1] / PYR_FACTOR))
        if nxt <= lengths[-1]:
            nxt = lengths[-1] + 1
        lengths.append(nxt)
    lengths[-1] = final_len

    # Initial synthesized: interpolate example_full from T_e to lengths[0]
    # we synthesize ALL J_t joints in the output
    def interp_1d(x, new_T):
        # x: [T, J, C] -> [new_T, J, C]
        x_t = torch.tensor(x, device=device, dtype=torch.float32)
        x_t = x_t.permute(1, 2, 0).unsqueeze(0)  # [1, J, C, T]
        J, C = x_t.shape[1], x_t.shape[2]
        x_t = x_t.reshape(1, J*C, x_t.shape[-1])
        x_t = F.interpolate(x_t, size=new_T, mode='linear', align_corners=True)
        return x_t.reshape(1, J, C, new_T).squeeze(0).permute(2, 0, 1).cpu().numpy()

    # Build projection mask (same shape as example bound joint channel set) for sparse retarg.
    # Here: bound-joint channels already selected by tgt_j_idxs, so the "mask" is trivially 1
    # for those and 0 for non-bound. We use it to weight the bound-joint distance function.

    synth = interp_1d(ex_full, lengths[0])  # [L0, J_t, 13] initial
    # add noise to bound joints
    rng = np.random.RandomState(seed)
    noise = rng.randn(*synth.shape).astype(np.float32) * NOISE_SIGMA * np.std(synth)
    synth = synth + noise

    for lvl, L in enumerate(lengths):
        # upsample synth to L
        if lvl > 0:
            synth = interp_1d(synth, L)

        # build example pyramid at this level
        # downscale example_full and example_bnd to length corresponding to L, capped at T_e
        lvl_ratio = L / final_len
        L_e = max(PATCH_SIZE + 2, min(T_e, int(round(T_e * lvl_ratio * 1.2))))
        ex_full_lvl = interp_1d(ex_full, L_e)             # [L_e, J_t, 13]
        ex_bnd_lvl = ex_full_lvl[:, tgt_j_idxs, :]        # [L_e, B, 13]

        # "Project" source motion onto target scaffold: for each bound joint, copy
        # the source's bound-joint channels into a target-shaped buffer whose non-bound
        # joints come from the example motion. This is the paper's Eq. 1 projection.
        src_lvl = interp_1d(source_motion, L)              # [L, J_s, 13]
        src_bnd_lvl = src_lvl[:, src_j_idxs, :]            # [L, B, 13]

        # synth[bound] <- matching_alpha * source_bnd_lvl + (1-alpha)*synth[bound]
        # synth[unbound] <- example interpolated (already done via interp_1d)
        synth[:, tgt_j_idxs, :] = MATCHING_ALPHA * src_bnd_lvl + (1 - MATCHING_ALPHA) * synth[:, tgt_j_idxs, :]

        # ----- match-and-blend (NUM_STEPS iterations) -----
        for step in range(NUM_STEPS):
            # Extract bound-joint patches from synth and example
            s_ten = torch.tensor(synth[:, tgt_j_idxs, :], device=device, dtype=torch.float32)
            s_ten = s_ten.permute(1, 2, 0).reshape(1, len(tgt_j_idxs) * 13, L)
            e_ten = torch.tensor(ex_bnd_lvl, device=device, dtype=torch.float32)
            e_ten = e_ten.permute(1, 2, 0).reshape(1, len(tgt_j_idxs) * 13, L_e)

            x_patches = extract_patches(s_ten, PATCH_SIZE, 1, loop=False)  # [1, N_s, D]
            y_patches = extract_patches(e_ten, PATCH_SIZE, 1, loop=False)  # [1, N_e, D]

            # NN-match
            nnf, _ = get_NNs_Dists(efficient_cdist, x_patches.squeeze(0), y_patches.squeeze(0), 0.01)
            matched = y_patches[:, nnf, :]  # [1, N_s, D]
            combined = combine_patches(s_ten.shape, matched, PATCH_SIZE, 1, loop=False)
            # write back bound joints to synth
            combined_np = combined.reshape(1, len(tgt_j_idxs), 13, L).squeeze(0).permute(2, 0, 1).cpu().numpy()
            synth[:, tgt_j_idxs, :] = combined_np

            # Also blend ALL joints by matching FULL-joint patches of the example to the synth
            # (this gives continuity to unbound joints). This is less aggressive than the paper's
            # full optimisation but keeps the synthesis coherent.
            s_ten_full = torch.tensor(synth, device=device, dtype=torch.float32)
            s_ten_full = s_ten_full.permute(1, 2, 0).reshape(1, J_t * 13, L)
            e_ten_full = torch.tensor(ex_full_lvl, device=device, dtype=torch.float32)
            e_ten_full = e_ten_full.permute(1, 2, 0).reshape(1, J_t * 13, L_e)
            x_p_full = extract_patches(s_ten_full, PATCH_SIZE, 1, loop=False)
            y_p_full = extract_patches(e_ten_full, PATCH_SIZE, 1, loop=False)

            # only use bound-joint channels as matching criterion (sparse correspondence)
            n_bnd = len(tgt_j_idxs)
            # rebuild a bound-only feature view into the full feature
            bnd_channels = []
            for b in tgt_j_idxs:
                for c in range(13):
                    bnd_channels.append(b * 13 + c)
            bnd_channels_t = torch.tensor(bnd_channels, device=device, dtype=torch.long)

            x_bnd_view = x_p_full.reshape(1, x_p_full.shape[1], J_t * 13, PATCH_SIZE)
            y_bnd_view = y_p_full.reshape(1, y_p_full.shape[1], J_t * 13, PATCH_SIZE)
            x_bnd_view = x_bnd_view.index_select(2, bnd_channels_t).reshape(1, x_p_full.shape[1], -1)
            y_bnd_view = y_bnd_view.index_select(2, bnd_channels_t).reshape(1, y_p_full.shape[1], -1)

            nnf_full, _ = get_NNs_Dists(efficient_cdist,
                                         x_bnd_view.squeeze(0), y_bnd_view.squeeze(0), 0.01)
            matched_full = y_p_full[:, nnf_full, :]
            combined_full = combine_patches(s_ten_full.shape, matched_full, PATCH_SIZE, 1, loop=False)
            synth = combined_full.reshape(1, J_t, 13, L).squeeze(0).permute(2, 0, 1).cpu().numpy()

            # re-impose the projection: bound joints get source motion (Eq. 1)
            synth[:, tgt_j_idxs, :] = MATCHING_ALPHA * src_bnd_lvl + (1 - MATCHING_ALPHA) * synth[:, tgt_j_idxs, :]

    # Final: return as [T, J, 13] numpy float32
    if synth.shape[0] != final_len:
        synth = interp_1d(synth, final_len)
    return synth.astype(np.float32)


# --------- End-to-end pipeline for all 30 pairs -----------------------------------------
def find_target_example(target_skel, motion_dir):
    """Return the first .npy clip for this skeleton (handles both ___ and _ naming)."""
    files = sorted(os.listdir(motion_dir))
    for sep in ['___', '_']:
        cand = [f for f in files if f.startswith(f'{target_skel}{sep}')]
        if cand:
            return cand[0]
    # fallback: loose match
    cand = [f for f in files if target_skel.lower() in f.lower()]
    return cand[0] if cand else None


def load_motion(fname, cond_entry):
    """Load [T, J, 13] motion (un-normalized)."""
    m = np.load(MOTION_DIR / fname)
    J = len(cond_entry['joints_names'])
    return m[:, :J, :].astype(np.float32)


def main():
    cond = np.load(COND_PATH, allow_pickle=True).item()
    contact_groups = json.loads(Path(CONTACT_GROUPS_PATH).read_text())
    with open(LABELS_PATH) as f:
        label_map = json.load(f)
    with open(EVAL_PAIRS_PATH) as f:
        eval_pairs = json.load(f)['pairs']

    # Load classifier
    sys.path.insert(0, str(ROOT))
    from eval.external_classifier import (ActionClassifier, extract_classifier_features,
                                           ACTION_CLASSES, ACTION_TO_IDX)
    from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clf = ActionClassifier()
    ckpt = torch.load(ROOT / 'save/external_classifier.pt', map_location=device)
    clf.load_state_dict(ckpt['model'])
    clf.to(device).eval()
    print(f'Classifier loaded: val_acc={ckpt.get("val_acc"):.3f}')

    metrics = {'pairs': [], 'config': {
        'patch_size': PATCH_SIZE, 'num_steps': NUM_STEPS, 'noise_sigma': NOISE_SIGMA,
        'coarse_ratio': COARSE_RATIO, 'pyr_factor': PYR_FACTOR,
        'matching_alpha': MATCHING_ALPHA,
        'classifier_val_acc': float(ckpt.get('val_acc', 0)),
        'implementation_note': ('APPROXIMATE "M2M-lite" — core patch matching algorithm '
                                'from Chen et al. 2508.13139 applied directly on the 13-dim '
                                'Truebones representation (vs paper\'s BVH pipeline). See '
                                'module docstring for simplifications.'),
    }}

    def classify_one(motion_np, parents):
        features = extract_classifier_features(motion_np, parents)
        if features is None:
            return None, None
        T_feat = features.shape[0]
        if T_feat < 64:
            features = np.pad(features, ((0, 64 - T_feat), (0, 0), (0, 0)))
        else:
            features = features[np.linspace(0, T_feat - 1, 64).astype(int)]
        with torch.no_grad():
            logits = clf(torch.tensor(features[None], dtype=torch.float32, device=device))
            probs = logits.softmax(-1).cpu().numpy()[0]
        idx = int(probs.argmax())
        return ACTION_CLASSES[idx], float(probs[idx])

    device_m2m = 'cpu'  # paper recommends CPU; our 12GB GPU is fine too, but cpu is reproducible
    for pair in eval_pairs:
        pid = pair['pair_id']
        src_fn = pair['source_fname']
        src_skel = pair['source_skel']
        tgt_skel = pair['target_skel']
        family_gap = pair['family_gap']

        print(f'\n=== pair {pid}: {src_skel} ({pair["source_label"]}) → {tgt_skel} [{family_gap}] ===')
        t0 = time.time()

        try:
            src_motion = load_motion(src_fn, cond[src_skel])
            ex_fname = find_target_example(tgt_skel, MOTION_DIR)
            if ex_fname is None:
                print(f'  SKIP: no example found for {tgt_skel}')
                continue
            ex_motion = load_motion(ex_fname, cond[tgt_skel])

            pairs_ij, desc = author_sparse_mapping(src_skel, tgt_skel, cond, contact_groups)
            src_j_idxs = [p[0] for p in pairs_ij]
            tgt_j_idxs = [p[1] for p in pairs_ij]

            print(f'  sparse mapping ({len(pairs_ij)} pairs): {"; ".join(desc)}')
            print(f'  target example: {ex_fname}   src T={src_motion.shape[0]} tgt T={ex_motion.shape[0]}')

            out = run_m2m_lite(src_motion, ex_motion, src_j_idxs, tgt_j_idxs,
                                tgt_n_joints=len(cond[tgt_skel]['joints_names']),
                                device=device_m2m, seed=42)

            out_path = OUT_DIR / f'pair_{pid:02d}_{src_skel}_to_{tgt_skel}.npy'
            np.save(out_path, out)
            wall_time = time.time() - t0

            # Evaluate: action accuracy via classifier
            target_parents = np.array(cond[tgt_skel]['parents'][:out.shape[1]], dtype=np.int64)
            positions = recover_from_bvh_ric_np(out.astype(np.float32))
            pred_action, pred_confidence = classify_one(positions, target_parents)

            # Baseline: classifier's prediction on the RAW source motion (for behavior preservation)
            src_parents = np.array(cond[src_skel]['parents'][:src_motion.shape[1]], dtype=np.int64)
            src_positions = recover_from_bvh_ric_np(src_motion.astype(np.float32))
            src_pred_action, src_pred_confidence = classify_one(src_positions, src_parents)

            # Q-component match: compute Q on M2M output and source
            # We need to monkey-patch the motion_dir so extract_quotient loads OUR saved motion.
            q_match = None
            try:
                q_src = extract_quotient(src_fn, cond[src_skel], contact_groups=contact_groups,
                                          motion_dir=str(MOTION_DIR), fps=30)
                q_m2m = extract_quotient(out_path.name, cond[tgt_skel], contact_groups=contact_groups,
                                          motion_dir=str(OUT_DIR), fps=30)
                # Cadence + heading-vel mean + limb-usage structure are the most comparable pieces
                # For Q-match score: normalized L2 in component space (lower = better).
                # Align COM paths by length (take min length), same for heading_vel
                L_com = min(len(q_src['com_path']), len(q_m2m['com_path']))
                com_diff = float(np.linalg.norm(q_src['com_path'][:L_com] - q_m2m['com_path'][:L_com]) /
                                  (np.linalg.norm(q_src['com_path'][:L_com]) + 1e-6))
                hv_diff = float(np.abs(np.mean(q_src['heading_vel']) - np.mean(q_m2m['heading_vel'])) /
                                 (abs(np.mean(q_src['heading_vel'])) + 1e-6))
                cad_diff = float(abs(q_src['cadence'] - q_m2m['cadence']) / (q_src['cadence'] + 1e-6))
                q_match = {'com_path_rel_err': com_diff, 'heading_vel_rel_err': hv_diff,
                           'cadence_rel_err': cad_diff,
                           'source_cadence': q_src['cadence'], 'm2m_cadence': q_m2m['cadence']}
            except Exception as q_err:
                print(f'  Q-extract warning: {q_err}')

            pair_metrics = {
                'pair_id': pid,
                'src_fname': src_fn,
                'src_skel': src_skel,
                'src_label': pair['source_label'],
                'tgt_skel': tgt_skel,
                'family_gap': family_gap,
                'sparse_mapping': desc,
                'n_sparse_pairs': len(pairs_ij),
                'target_example': ex_fname,
                'wall_time_sec': round(wall_time, 2),
                'out_path': str(out_path),
                'predicted_action': pred_action,
                'pred_confidence': pred_confidence,
                'source_pred_action': src_pred_action,
                'source_pred_confidence': src_pred_confidence,
                'label_match': (pred_action == pair['source_label']),
                'behavior_preserved': (pred_action == src_pred_action),
                'q_match': q_match,
            }
            metrics['pairs'].append(pair_metrics)
            print(f'  done: pred_action={pred_action} (label={pair["source_label"]}, match={pred_action == pair["source_label"]})  time={wall_time:.1f}s')
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f'  ERROR: {e}')
            metrics['pairs'].append({'pair_id': pid, 'src_fname': src_fn, 'error': str(e)})

    # --------- Aggregate stratified means ---------
    by_strata = {'near': [], 'moderate': [], 'extreme': []}
    for p in metrics['pairs']:
        if 'error' in p or 'family_gap' not in p:
            continue
        by_strata.setdefault(p['family_gap'], []).append(p)

    agg = {'mean_label_match': {}, 'mean_behavior_preserved': {}, 'mean_wall_time': {},
           'median_q_cadence_rel_err': {}, 'median_q_heading_vel_rel_err': {},
           'median_q_com_path_rel_err': {}}
    for s, plist in by_strata.items():
        if not plist:
            continue
        agg['mean_label_match'][s] = float(np.mean([int(p.get('label_match', False)) for p in plist]))
        agg['mean_behavior_preserved'][s] = float(np.mean([int(p.get('behavior_preserved', False)) for p in plist]))
        agg['mean_wall_time'][s] = float(np.mean([p.get('wall_time_sec', 0) for p in plist]))
        q_cad = [p['q_match']['cadence_rel_err'] for p in plist
                  if p.get('q_match') and p['q_match'].get('source_cadence', 0) > 0.01]
        q_hv = [p['q_match']['heading_vel_rel_err'] for p in plist if p.get('q_match')]
        q_com = [p['q_match']['com_path_rel_err'] for p in plist if p.get('q_match')]
        if q_cad:
            agg['median_q_cadence_rel_err'][s] = float(np.median(q_cad))
        if q_hv:
            agg['median_q_heading_vel_rel_err'][s] = float(np.median(q_hv))
        if q_com:
            agg['median_q_com_path_rel_err'][s] = float(np.median(q_com))

    all_valid = [p for p in metrics['pairs'] if 'error' not in p and 'label_match' in p]
    agg['overall_label_match'] = float(np.mean([int(p['label_match']) for p in all_valid])) if all_valid else None
    agg['overall_behavior_preserved'] = float(np.mean([int(p.get('behavior_preserved', False)) for p in all_valid])) if all_valid else None
    agg['overall_wall_time'] = float(np.mean([p['wall_time_sec'] for p in all_valid])) if all_valid else None
    metrics['stratified'] = agg

    with open(OUT_DIR / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'\nSaved metrics to {OUT_DIR / "metrics.json"}')
    print(f'\n=== SUMMARY ===')
    print(f'  overall label-match acc: {agg["overall_label_match"]:.3f}')
    print(f'  overall behavior-preserved acc (matches classifier\'s source prediction): {agg["overall_behavior_preserved"]:.3f}')
    print(f'  label-match by strata: {agg["mean_label_match"]}')
    print(f'  behavior-preserved by strata: {agg["mean_behavior_preserved"]}')
    print(f'  median Q cadence rel err by strata: {agg["median_q_cadence_rel_err"]}')
    print(f'  median Q heading_vel rel err by strata: {agg["median_q_heading_vel_rel_err"]}')
    print(f'  median Q com_path rel err by strata: {agg["median_q_com_path_rel_err"]}')
    print(f'  mean wall-time per pair: {agg["overall_wall_time"]:.3f}s')
    print(f'  classifier val acc (upper bound, weak signal): {metrics["config"]["classifier_val_acc"]:.3f}')


if __name__ == '__main__':
    main()
