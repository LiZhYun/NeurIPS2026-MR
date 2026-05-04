"""Minimal-Correspondence Escape (Codex go/no-go pilot, 2026-04-14).

Research question (Codex): If we inject 2-6 manually-authored paired bone
correspondences per hard support-absent pair, plus one or a few target-skeleton
"semantically-closest" anchor motions, can a tiny sparse-correspondence-guided
retrieval + temporal smoothing pipeline lift support-absent label-match from
the current 0.083 ceiling to >= 0.25-0.30?

Approach (fallback-friendly):
  For each of 6 authored absent pairs:
    1. Load source motion [T_s, J_s, 13].
    2. Load 1-3 pre-selected anchor target-skel clips (cross-available ACTION
       that is SEMANTICALLY closest to source's label). Concatenate into a
       single pool of anchor frames [T_a, J_t, 13].
    3. Using the authored sparse bone correspondences (2-6 pairs), compute a
       per-frame pose feature consisting of:
        - Root trajectory (source COM re-scaled by body-scale ratio).
        - Bound-joint root-relative POSITIONS (the 3-dim pos channel of source
          at src_j_idxs).
        - Bound-joint foot-contact BIT.
       For each source frame, find the nearest anchor frame by L2 on these
       sparse-bone features (using paired target joints tgt_j_idxs).
    4. Build the retargeted clip: for each source frame t, take the matched
       anchor frame at the paired target joints; fill non-paired joints from
       the same anchor frame (so the whole skeleton is consistent).
    5. Temporal smoothing: Hanning-window smooth over a +/- 3-frame window to
       avoid retrieval chatter.
    6. Overwrite bound-joint contact_channel (ch 12) with source's contact at
       the paired-bone joints, so the target clip carries the source's foot-
       contact rhythm — this is the "Motion2Motion projection" constraint
       restricted to authored correspondences.
    7. Save [T_s, J_t, 13] and compute classifier-v2 label_match +
       behavior_preserved + contact_f1_vs_source + Q-components.

This is the "retrieval-with-paired-bones" baseline (no trained adapter).
If this alone fails the thresholds, the adapter story is dead.

Usage:
    conda run -n anytop python eval/run_minimal_corr.py

Outputs:
    eval/results/k_compare/minimal_corr/pair_<id>_<src>_to_<tgt>.npy
    eval/results/k_compare/minimal_corr/metrics.json
"""
from __future__ import annotations

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(str(PROJECT_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MOTION_DIR = ROOT / 'dataset/truebones/zoo/truebones_processed/motions'
COND_PATH = ROOT / 'dataset/truebones/zoo/truebones_processed/cond.npy'
CONTACT_GROUPS_PATH = ROOT / 'eval/quotient_assets/contact_groups.json'
LABELS_PATH = ROOT / 'dataset/truebones/zoo/truebones_processed/motion_labels.json'
EVAL_PAIRS_PATH = ROOT / 'idea-stage/eval_pairs.json'
OUT_DIR = ROOT / 'eval/results/k_compare/minimal_corr'
OUT_DIR.mkdir(parents=True, exist_ok=True)
CLF_V2 = ROOT / 'save/external_classifier_v2.pt'

FOOT_CH_IDX = 12
POS_SLICE = slice(0, 3)
ROT_SLICE = slice(3, 9)
VEL_SLICE = slice(9, 12)
SMOOTH_WIN = 3  # +/- frames


# =============================================================================
# Authored pair correspondences. For each of 6 absent pairs, hand-author:
#   - source_idxs: indices into source skeleton's joints_names (2-6 entries)
#   - target_idxs: paired indices into target skeleton's joints_names
#   - anchor_clips: semantically-closest target-skel motions (1-3 files)
#   - rationale: short note on what is paired and why
# =============================================================================
#
# Rationale:
#   pid=11 Ostrich(walk)->Crocodile [moderate]: biped bird vs. sprawling lizard
#     both walk on ground; map hind-feet(ostrich) <-> hind-feet(croc),
#     neck-tip(ostrich head BN_HeadNub 52) <-> head(croc 33). Use Crocodile
#     Running clip as the anchor (locomotion proxy).
#   pid=12 Comodoa(attack)->Puppy [moderate]: both quadrupeds; direct limb map
#     (LF<->LF, RF<->RF, LH<->LH, RH<->RH). Attack isn't available on puppy,
#     so use Puppy_Run + Puppy_IdleEnergetic as anchor pool — both carry the
#     pouncing-like dynamic energy. Also pair head(Comodoa 54)<->head(Puppy 32).
#   pid=15 Goat(eat)->Ant [extreme]: head/mouth eating on quadruped -> mouth on
#     insect. Hand=tip of forelimb of goat (no extensive chewing). Map head/jaw
#     (Goat head 5, Goat mouth 6) <-> (Ant head 36, Ant mouth_L 37). Plus
#     forelimb tips for body reference. Anchor = Ant idle (no eat in Ant).
#   pid=17 Parrot(fly)->PolarBearB [moderate]: wing flap on bird -> forelimb
#     flail on quadruped. Map wing roots (Parrot 42 L_Wing_01, 63 R_Wing_01)
#     <-> clavicles (BearB 16 R_Clavicle, 22 L_Clavicle). Plus wing-tip (Parrot
#     45 L_Wing_04, 68 R_Wing_03) <-> hand (BearB 19 R_Hand, 25 L_Hand).
#     Anchor = PolarBearB_GetUp + Idle (most wing-flap-like arm motion).
#   pid=19 Bird(idle)->Pteranodon [moderate]: both bipeds; body pose is close.
#     Map pelvis, spine, head, R_foot, L_foot in both skeletons. Anchor =
#     Pteranodon flapergasted (closest still-with-breathing to idle).
#   pid=27 Buzzard(fly)->Centipede [extreme]: wings on bird <-> leg-wave on
#     centipede. Best we can do: map root and tail endpoints since the action
#     is dominated by COM/body-axis motion, and tail segments are closest to
#     the centipede body axis. Pair wing-tip (Buzzard 44 R_Wing_04, 51 L_Wing_04)
#     <-> outermost leg pairs (Cent 9 R_Foot_01, 4 L_Foot_01). Anchor =
#     Centipede Run (locomotion baseline).

AUTHORED_PAIRS = {
    11: {  # Ostrich(walk) -> Crocodile [moderate]
        'source_idxs': [0, 5, 9, 19, 48, 52],
        'target_idxs': [0, 6, 9, 14, 33, 37],
        'correspondence_desc': [
            'root(Bip01_Pelvis)<->root(Bip01_Pelvis)',
            'spine(Ostrich Bip01_Spine)<->spine(Croc Bip01_Spine)',
            'R_Foot(Ostrich Bip01_R_Foot)<->R_Foot(Croc Bip01_R_Foot)',
            'L_Foot(Ostrich Bip01_L_Foot)<->L_Foot(Croc Bip01_L_Foot)',
            'Head(Ostrich Bip01_Head)<->Head(Croc Bip01_Head)',
            'HeadNub(Ostrich Bip01_HeadNub)<->HeadNub(Croc Bip01_HeadNub)',
        ],
        'rationale': (
            'Biped bird -> sprawling quadruped lizard. Both walk on ground; '
            'hind-feet, spine, and head are direct analogues (6 pairs).'
        ),
        'anchor_clips': ['Crocodile___Running_255.npy', 'Crocodile___Idle_258.npy'],
    },
    12: {  # Comodoa(attack) -> Puppy [moderate]
        'source_idxs': [0, 52, 42, 10, 20, 54],
        'target_idxs': [0, 30, 24, 16, 10, 32],
        'correspondence_desc': [
            'root<->root',
            'L_forelimb_tip(Comodoa Bip01_L_Finger0Nub)<->L_Hand_tip(Puppy Bip01_L_Finger0Nub)',
            'R_forelimb_tip(Comodoa Bip01_R_Finger0Nub)<->R_Hand_tip(Puppy Bip01_R_Finger0Nub)',
            'L_toe_tip(Comodoa Bip01_L_Toe0Nub)<->L_toe_tip(Puppy Bip01_L_Toe0Nub)',
            'R_toe_tip(Comodoa Bip01_R_Toe0Nub)<->R_toe_tip(Puppy Bip01_R_Toe0Nub)',
            'head(Comodoa Bip01_Head)<->head(Puppy Bip01_Head)',
        ],
        'rationale': (
            'Both quadrupeds; direct limb and head analogues. Attack involves '
            'forelimb strike + head lunge.'
        ),
        'anchor_clips': ['Puppy_Puppy_Run_665.npy', 'Puppy_Puppy_IdleEnergetic_666.npy'],
    },
    15: {  # Goat(eat) -> Ant [extreme]
        'source_idxs': [0, 5, 6, 11, 17, 23],
        'target_idxs': [0, 36, 37, 34, 27, 18],
        'correspondence_desc': [
            'root<->root',
            'head(Goat Bip01_Head)<->head(Ant Bip01_Head)',
            'mouth(Goat BN_Mouth_01)<->mouth_L(Ant BN_Mouth_L_01)',
            'L_forelimb_tip(Goat Bip01_L_Finger0)<->L_forelimb_tip(Ant Bip01_L_Finger01_end_site)',
            'R_forelimb_tip(Goat Bip01_R_Finger0)<->R_forelimb_tip(Ant Bip01_R_Finger01)',
            'L_hind_toe(Goat Bip01_L_Toe0)<->L_hind_toe(Ant Bip01_L_Toe0)',
        ],
        'rationale': (
            'Cross-family (quadruped mammal -> 6-leg insect). Eat = head/mouth '
            'motion + some standing. Map head/mouth directly; forelimbs and '
            'hind-limb tips provide body-scale reference.'
        ),
        'anchor_clips': ['Ant___Idle_45.npy', 'Ant___Idle2_54.npy'],
    },
    17: {  # Parrot(fly) -> PolarBearB [moderate]
        'source_idxs': [0, 42, 63, 45, 68, 51],
        'target_idxs': [0, 22, 16, 25, 19, 29],
        'correspondence_desc': [
            'root<->root',
            'L_Wing_root(Parrot BN_Wing_L_01)<->L_Clavicle(BearB Bip01_L_Clavicle)',
            'R_Wing_root(Parrot BN_Wing_R_01)<->R_Clavicle(BearB Bip01_R_Clavicle)',
            'L_Wing_tip(Parrot BN_Wing_L_04)<->L_Hand(BearB Bip01_L_Hand)',
            'R_Wing_tip(Parrot BN_Wing_R_03)<->R_Hand(BearB Bip01_R_Hand)',
            'head(Parrot Bip01_Head)<->head(BearB Bip01_Head)',
        ],
        'rationale': (
            'Flying bird -> walking quadruped. Improvised: wing flap <-> '
            'forelimb flail. Map wing root to shoulder and tip to hand to '
            'preserve flapping energy profile.'
        ),
        'anchor_clips': ['PolarBearB___GetUp_645.npy', 'PolarBearB___Idle_643.npy'],
    },
    19: {  # Bird(idle) -> Pteranodon [moderate]
        'source_idxs': [0, 54, 34, 58, 59, 4],
        'target_idxs': [0, 7, 18, 30, 31, 3],
        'correspondence_desc': [
            'root(Bird Bip01_Pelvis)<->root(Pter Hips)',
            'L_toe(Bird Bip01_L_Toe0Nub)<->L_foot(Pter jt_Foot_L)',
            'R_toe(Bird Bip01_R_Toe0Nub)<->R_foot(Pter jt_Foot_R)',
            'head(Bird Bip01_Head)<->head(Pter jt_Head_C)',
            'mouth(Bird BN_Mouth_01)<->jaw(Pter jt_Jaw_C)',
            'spine(Bird Bip01_Spine)<->spine(Pter jt_Spine2_C)',
        ],
        'rationale': (
            'Both small-to-medium bipeds with similar body axis during idle. '
            'Map spine, head, and foot tips directly; skeleton body layout '
            'matches. Flying creature in idle mode (flapergasted) is closest '
            'visual match.'
        ),
        'anchor_clips': ['Pteranodon___Flapergasted_653.npy', 'Pteranodon___FlyLoop_657.npy'],
    },
    27: {  # Buzzard(fly) -> Centipede [extreme]
        'source_idxs': [0, 44, 51, 60, 38, 22],
        'target_idxs': [0, 5, 34, 76, 36, 11],
        'correspondence_desc': [
            'root<->root',
            'R_Wing_tip(Buzzard BN_Wing_R_04)<->L_Toe0_01(Cent BN_Toe0_L_01)',
            'L_Wing_tip(Buzzard BN_Wing_L_04)<->L_Toe0(Cent Bip01_L_Toe0)',
            'head(Buzzard Bip01_HeadNub)<->headNub(Cent Bip01_HeadNub)',
            'L_hind_toe(Buzzard Bip01_L_Toe0Nub)<->L_rear_foot(Cent BN_Foot_L_02)',
            'R_hind_toe(Buzzard Bip01_R_Toe0Nub)<->R_front_mid_foot(Cent Bip01_R_Toe0Nub_approx via idx 11)',
        ],
        'rationale': (
            'Most extreme pair: flying bird (60 joints) -> 83-joint multi-segment '
            'centipede. No meaningful anatomical analogue for wings; improvise '
            'wing-tips <-> outermost segment feet to map the lateral energy. '
            'Map head tip <-> head tip; hind toes to rear segment feet. '
            'Anchor = Centipede Run (locomotion carrier).'
        ),
        'anchor_clips': ['Centipede___Run_206.npy', 'Centipede___Idle_200.npy'],
    },
}


# =============================================================================
# I/O + helpers
# =============================================================================
def load_motion(fname, cond_entry):
    m = np.load(MOTION_DIR / fname)
    J = len(cond_entry['joints_names'])
    return m[:, :J, :].astype(np.float32)


def denormalize_motion(m_norm, cond_entry):
    """If motion is z-scored (|m|<5 heuristic), multiply by std and add mean."""
    J = m_norm.shape[1]
    std = cond_entry['std'][:J]
    mean = cond_entry['mean'][:J]
    if np.abs(m_norm).max() < 5:
        return m_norm * std + mean
    return m_norm


def normalize_motion(m_phys, cond_entry):
    J = m_phys.shape[1]
    std = cond_entry['std'][:J]
    mean = cond_entry['mean'][:J]
    return (m_phys - mean) / std


def body_scale(cond_entry):
    return float(np.linalg.norm(cond_entry['offsets'], axis=1).sum() + 1e-6)


# =============================================================================
# Core: anchor-conditioned sparse-correspondence retrieval
# =============================================================================
def build_sparse_pose_feature(motion_phys, joint_idxs, body_scale_hint=1.0):
    """Extract rich per-frame feature from bound joints.

    Per bound joint: root-relative position (3) + velocity (3) + contact (1)
    plus a temporal first-difference of position (3). These give the feature
    a dynamic fingerprint (not just the pose) so NN retrieval does not
    collapse onto the lowest-magnitude anchor frame.
    """
    T = motion_phys.shape[0]
    feats = []
    for j in joint_idxs:
        if j >= motion_phys.shape[1]:
            feats.append(np.zeros((T, 10), dtype=np.float32))
            continue
        pos = motion_phys[:, j, POS_SLICE] / max(body_scale_hint, 1e-6)  # scale-aware
        vel = motion_phys[:, j, VEL_SLICE]
        contact = motion_phys[:, j, FOOT_CH_IDX:FOOT_CH_IDX + 1]
        dpos = np.diff(pos, axis=0, prepend=pos[:1])
        feats.append(np.concatenate([pos, vel, contact, dpos], axis=-1))
    feat = np.concatenate(feats, axis=-1).astype(np.float32)
    return feat


def build_global_dynamic_signature(motion_phys, joint_idxs, body_scale_hint=1.0,
                                    window=5):
    """Feature: per-frame ENERGY + contact-bits over bound joints, smoothed.

    Rough idea: 'how much are the bound joints moving at frame t?'. This should
    align source and anchor by activity level (flying wings move a lot;
    standing still does not), which is the dominant signal the classifier
    should see.
    """
    T = motion_phys.shape[0]
    feats = np.zeros((T, 3), dtype=np.float32)
    for j in joint_idxs:
        if j >= motion_phys.shape[1]:
            continue
        v = motion_phys[:, j, VEL_SLICE]  # [T, 3]
        feats[:, 0] += np.linalg.norm(v, axis=-1) / max(body_scale_hint, 1e-6)
        feats[:, 1] += motion_phys[:, j, FOOT_CH_IDX]
        feats[:, 2] += np.linalg.norm(
            np.diff(motion_phys[:, j, POS_SLICE], axis=0, prepend=motion_phys[:1, j, POS_SLICE]),
            axis=-1,
        ) / max(body_scale_hint, 1e-6)
    if window > 1:
        # moving-average smoothing for robustness
        w = np.hanning(window)
        w = w / w.sum()
        pad = window // 2
        feats_padded = np.pad(feats, ((pad, pad), (0, 0)), mode='edge')
        feats_smooth = np.zeros_like(feats)
        for t in range(T):
            feats_smooth[t] = (feats_padded[t:t + window] * w[:, None]).sum(0)
        return feats_smooth
    return feats


def nn_match_per_frame(src_feat, anchor_feat, monotonicity_penalty=0.0):
    """For each source frame, return its nearest-anchor-frame index.

    The two feature tensors must have the SAME last dim and should ALREADY
    be scale-normalized by the caller. If monotonicity_penalty > 0, we add
    a soft cost that penalizes non-monotone frame hops, which prevents the
    retrieval from collapsing to a single anchor frame.
    """
    T_s, D = src_feat.shape
    T_a = anchor_feat.shape[0]

    # z-score each feature dim over the concatenated pool so distances are
    # balanced across channels.
    pool = np.concatenate([src_feat, anchor_feat], axis=0)
    mu = pool.mean(axis=0, keepdims=True)
    sd = pool.std(axis=0, keepdims=True) + 1e-3
    src_n = ((src_feat - mu) / sd).astype(np.float32)
    anc_n = ((anchor_feat - mu) / sd).astype(np.float32)

    src_t = torch.from_numpy(src_n).float()
    anc_t = torch.from_numpy(anc_n).float()
    if torch.cuda.is_available():
        src_t = src_t.cuda(); anc_t = anc_t.cuda()
    s2 = (src_t ** 2).sum(-1, keepdim=True)
    a2 = (anc_t ** 2).sum(-1, keepdim=True).t()
    d2 = s2 + a2 - 2 * src_t @ anc_t.t()
    d2 = d2.clamp_min(0.0)
    # Hop penalty: cost(t, a) += lambda * (a / T_a - t / T_s)^2 * D
    if monotonicity_penalty > 0:
        t_axis = torch.arange(T_s, device=d2.device).float().unsqueeze(-1) / max(T_s - 1, 1)
        a_axis = torch.arange(T_a, device=d2.device).float().unsqueeze(0) / max(T_a - 1, 1)
        d2 = d2 + monotonicity_penalty * ((a_axis - t_axis) ** 2) * D

    nnf = d2.argmin(dim=-1).cpu().numpy()
    dists = d2.gather(1, d2.argmin(dim=-1, keepdim=True)).sqrt().squeeze(-1).cpu().numpy()
    return nnf, dists


def hanning_smooth(arr, win=3):
    """Smooth along axis 0 with a Hann window of radius `win`."""
    if win <= 0:
        return arr
    T = arr.shape[0]
    w = np.hanning(2 * win + 1)
    w = w / w.sum()
    out = arr.copy()
    for t in range(T):
        lo = max(0, t - win)
        hi = min(T, t + win + 1)
        wi = w[max(0, win - (t - lo)):win + 1 + (hi - t - 1)]
        if wi.sum() <= 0:
            continue
        wi = wi / wi.sum()
        out[t] = (arr[lo:hi] * wi[:, None, None]).sum(axis=0)
    return out


def retarget_via_sparse_corr_retrieval(src_motion_phys, anchor_pool_phys,
                                       src_idxs, tgt_idxs,
                                       src_body_scale, tgt_body_scale,
                                       monotonicity=8.0, vel_splice_alpha=0.9):
    """Main algorithm: anchor-conditioned sparse-correspondence retrieval.

    Pipeline:
      (a) Build BOTH a "dynamic signature" feature (scalar energy + contact
          activity, smoothed) and a richer "sparse pose feature" (per-bound-
          joint pos/vel/contact/dpos).
      (b) Concatenate them -> z-score normalized -> do NN matching with a soft
          monotonicity penalty so we don't collapse to one anchor frame.
      (c) For each source frame t -> retrieved anchor pose, then SPLICE source
          velocity + source contact at paired-bone joints onto the anchor
          frame's corresponding target-bone slots. This is what gives the
          synthesized motion the source's action signature at the bound bones
          while keeping the target skeleton's natural pose for unbound joints.
      (d) Hanning temporal smoothing to kill retrieval chatter.

    Args:
        src_motion_phys: [T_s, J_s, 13]
        anchor_pool_phys: [T_a, J_t, 13]
        src_idxs, tgt_idxs: paired-bone indices (same length)
        monotonicity: soft cost for non-monotone hops across anchor pool.
        vel_splice_alpha: 0=retrieval only; 1=full source-velocity overwrite at
            paired-bone joints.

    Returns: [T_s, J_t, 13] retargeted motion, and nnf array.
    """
    T_s = src_motion_phys.shape[0]
    J_t = anchor_pool_phys.shape[1]

    # (a) Build features for matching
    src_pose = build_sparse_pose_feature(src_motion_phys, src_idxs, src_body_scale)
    anc_pose = build_sparse_pose_feature(anchor_pool_phys, tgt_idxs, tgt_body_scale)
    src_sig = build_global_dynamic_signature(src_motion_phys, src_idxs, src_body_scale)
    anc_sig = build_global_dynamic_signature(anchor_pool_phys, tgt_idxs, tgt_body_scale)
    # Concatenate; weight dynamic signature higher (it's the action discriminator)
    src_feat = np.concatenate([src_pose, 3.0 * src_sig], axis=-1)
    anc_feat = np.concatenate([anc_pose, 3.0 * anc_sig], axis=-1)

    # (b) NN match with monotonicity penalty
    nnf, _ = nn_match_per_frame(src_feat, anc_feat,
                                  monotonicity_penalty=monotonicity)

    # (c) Start from retrieved anchor poses, then HARD-PROJECT source bound-
    # bone features onto the target scaffold (Motion2Motion Eq. 1).
    retarget = anchor_pool_phys[nnf].copy()  # [T_s, J_t, 13]

    # Scale-transform source bound-bone positions to target body frame.
    scale_ratio = tgt_body_scale / max(src_body_scale, 1e-6)

    for si, ti in zip(src_idxs, tgt_idxs):
        if si >= src_motion_phys.shape[1] or ti >= retarget.shape[1]:
            continue
        # Alpha-blend the full position + 6D-rot channels + velocity + contact:
        # we trust the authored correspondence as an anatomical near-analogue.
        src_pos = src_motion_phys[:, si, POS_SLICE] * scale_ratio
        src_rot = src_motion_phys[:, si, ROT_SLICE]
        src_vel = src_motion_phys[:, si, VEL_SLICE] * scale_ratio
        src_con = src_motion_phys[:, si, FOOT_CH_IDX]
        alpha = vel_splice_alpha
        retarget[:, ti, POS_SLICE] = (
            (1 - alpha) * retarget[:, ti, POS_SLICE] + alpha * src_pos
        )
        retarget[:, ti, ROT_SLICE] = (
            (1 - alpha) * retarget[:, ti, ROT_SLICE] + alpha * src_rot
        )
        retarget[:, ti, VEL_SLICE] = (
            (1 - alpha) * retarget[:, ti, VEL_SLICE] + alpha * src_vel
        )
        retarget[:, ti, FOOT_CH_IDX] = src_con

    # Also copy the root DOF from source directly — the classifier uses root
    # trajectory heavily. Index 0 is always paired (we enforce pairs[0]==(0,0)).
    if src_idxs[0] == 0 and tgt_idxs[0] == 0:
        retarget[:, 0, POS_SLICE] = src_motion_phys[:, 0, POS_SLICE] * scale_ratio
        retarget[:, 0, VEL_SLICE] = src_motion_phys[:, 0, VEL_SLICE] * scale_ratio
        retarget[:, 0, ROT_SLICE] = src_motion_phys[:, 0, ROT_SLICE]

    # (d) Temporal smoothing
    retarget = hanning_smooth(retarget, win=SMOOTH_WIN)

    return retarget.astype(np.float32), nnf


# =============================================================================
# Evaluation: classifier-v2 + contact_f1_vs_source + Q-components
# =============================================================================
def classify_motion(clf, motion_phys, skel_cond, extract_fn, recover_fn):
    J_skel = skel_cond['offsets'].shape[0]
    m = motion_phys
    if m.shape[1] > J_skel:
        m = m[:, :J_skel]
    try:
        positions = recover_fn(m.astype(np.float32))
    except Exception:
        return None
    parents = skel_cond['parents'][:J_skel]
    feats = extract_fn(positions, parents)
    if feats is None or feats.shape[0] < 4:
        return None
    return clf.predict_label(feats)


def contact_f1(sched_rec: np.ndarray, sched_tgt: np.ndarray, thresh: float = 0.5) -> float:
    pred = (np.asarray(sched_rec) >= thresh).astype(np.int8).ravel()
    gt = (np.asarray(sched_tgt) >= thresh).astype(np.int8).ravel()
    n = min(pred.size, gt.size)
    pred, gt = pred[:n], gt[:n]
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    p = tp / (tp + fp + 1e-8); r = tp / (tp + fn + 1e-8)
    return float(2 * p * r / (p + r + 1e-8))


# =============================================================================
# Main
# =============================================================================
def main():
    cond = np.load(COND_PATH, allow_pickle=True).item()
    with open(CONTACT_GROUPS_PATH) as f:
        contact_groups = json.load(f)
    with open(EVAL_PAIRS_PATH) as f:
        eval_pairs = json.load(f)['pairs']
    pair_by_id = {p['pair_id']: p for p in eval_pairs}

    from eval.external_classifier import extract_classifier_features, ACTION_CLASSES
    from eval.train_external_classifier_v2 import V2Classifier
    from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np
    from eval.quotient_extractor import extract_quotient

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clf = V2Classifier(str(CLF_V2), device=device)
    print(f'Loaded classifier v2 (arch={clf.arch}) on {device}')

    results = {
        'config': {
            'method': 'minimal_correspondence_retrieval',
            'n_pairs': len(AUTHORED_PAIRS),
            'authored_pair_ids': sorted(AUTHORED_PAIRS.keys()),
            'smooth_win_frames': SMOOTH_WIN,
            'classifier_ckpt': str(CLF_V2),
        },
        'pairs': [],
    }

    t_all = time.time()
    for pid in sorted(AUTHORED_PAIRS.keys()):
        authored = AUTHORED_PAIRS[pid]
        meta = pair_by_id[pid]
        src_fname = meta['source_fname']; src_skel = meta['source_skel']
        tgt_skel = meta['target_skel']; src_label = meta['source_label']
        family_gap = meta['family_gap']
        support = int(meta['support_same_label'])
        print(f"\n=== pair {pid:02d}  {src_skel}({src_label}) -> {tgt_skel}  [{family_gap}, supp={support}] ===")

        rec = {
            'pair_id': pid, 'src_fname': src_fname, 'src_skel': src_skel,
            'src_label': src_label, 'tgt_skel': tgt_skel,
            'family_gap': family_gap, 'support_same_label': support,
            'authored_correspondences': authored['correspondence_desc'],
            'n_paired_bones': len(authored['source_idxs']),
            'rationale': authored['rationale'],
            'anchor_clips': authored['anchor_clips'],
        }
        t_pair = time.time()
        try:
            # Load source (may be normalized; denormalize to physical)
            src_norm = load_motion(src_fname, cond[src_skel])
            src_phys = denormalize_motion(src_norm, cond[src_skel])
            print(f'  src shape {src_phys.shape}')

            # Load and concatenate anchor clips (physical)
            anchor_list = []
            for af in authored['anchor_clips']:
                try:
                    a_norm = load_motion(af, cond[tgt_skel])
                    a_phys = denormalize_motion(a_norm, cond[tgt_skel])
                    anchor_list.append(a_phys)
                except Exception as e:
                    print(f'  WARN: anchor {af} load failed: {e}')
            if not anchor_list:
                raise RuntimeError('No anchor clips loaded')
            anchor_pool = np.concatenate(anchor_list, axis=0)
            print(f'  anchor pool shape {anchor_pool.shape} ({len(anchor_list)} clips)')

            src_bs = body_scale(cond[src_skel])
            tgt_bs = body_scale(cond[tgt_skel])
            rec['src_body_scale'] = src_bs
            rec['tgt_body_scale'] = tgt_bs
            rec['body_scale_ratio'] = tgt_bs / max(src_bs, 1e-6)

            # Retarget
            retarget_phys, nnf = retarget_via_sparse_corr_retrieval(
                src_phys, anchor_pool,
                authored['source_idxs'], authored['target_idxs'],
                src_bs, tgt_bs,
            )
            print(f'  retarget shape {retarget_phys.shape}  unique anchor frames used {len(set(nnf.tolist()))}/{len(nnf)}')

            # Save physical motion (same convention as motion2motion_run output)
            out_path = OUT_DIR / f'pair_{pid:02d}_{src_skel}_to_{tgt_skel}.npy'
            np.save(out_path, retarget_phys.astype(np.float32))
            rec['out_path'] = str(out_path)

            # Classify v2
            tgt_pred = classify_motion(
                clf, retarget_phys, cond[tgt_skel],
                extract_classifier_features, recover_from_bvh_ric_np,
            )
            src_pred = classify_motion(
                clf, src_phys, cond[src_skel],
                extract_classifier_features, recover_from_bvh_ric_np,
            )
            rec['tgt_pred'] = tgt_pred
            rec['src_classifier_pred'] = src_pred
            rec['label_match'] = bool(tgt_pred == src_label)
            rec['behavior_preserved'] = bool(src_pred is not None and tgt_pred == src_pred)
            print(f'  clf: tgt_pred={tgt_pred}  src_pred={src_pred}  '
                  f'label_match={rec["label_match"]}  '
                  f'behavior_preserved={rec["behavior_preserved"]}')

            # Q-components: write tmp file and use extract_quotient
            tmp_fname = f'__mc_pair_{pid:02d}.npy'
            tmp_path = MOTION_DIR / tmp_fname
            try:
                np.save(tmp_path, retarget_phys.astype(np.float32))
                q_src = extract_quotient(
                    src_fname, cond[src_skel],
                    contact_groups=contact_groups,
                    motion_dir=str(MOTION_DIR),
                )
                q_out = extract_quotient(
                    tmp_fname, cond[tgt_skel],
                    contact_groups=contact_groups,
                    motion_dir=str(MOTION_DIR),
                )
            finally:
                if tmp_path.exists():
                    try: tmp_path.unlink()
                    except Exception: pass

            # Contact F1 vs source (aggregate per-frame contact count)
            ss = np.asarray(q_src['contact_sched'])
            rs = np.asarray(q_out['contact_sched'])
            T = min(ss.shape[0], rs.shape[0])
            idx = np.clip(np.linspace(0, ss.shape[0] - 1, T).astype(int), 0, ss.shape[0] - 1)
            ss_t = ss[idx]
            ss_agg = ss_t.sum(axis=1) if ss_t.ndim == 2 else ss_t
            rs_agg = rs.sum(axis=1) if rs.ndim == 2 else rs
            rec['contact_f1_vs_source'] = contact_f1(rs_agg, ss_agg)

            # Q distance components
            def _l2(a, b):
                a = np.asarray(a); b = np.asarray(b)
                T_ = min(a.shape[0], b.shape[0])
                return float(np.linalg.norm((a[:T_] - b[:T_]).reshape(-1)))
            rec['q_com_path_l2'] = _l2(q_src['com_path'], q_out['com_path'])
            rec['q_heading_vel_l2'] = _l2(q_src['heading_vel'], q_out['heading_vel'])
            rec['q_cadence_abs_diff'] = float(abs(float(q_src['cadence']) - float(q_out['cadence'])))
            cs_src = ss_agg
            cs_out = rs_agg
            T_ = min(cs_src.size, cs_out.size)
            rec['q_contact_sched_l2'] = float(np.linalg.norm(cs_src[:T_] - cs_out[:T_]))

            rec['wall_time_s'] = float(time.time() - t_pair)
            rec['status'] = 'ok'

            print(f'  metrics: label_match={rec["label_match"]}  '
                  f'contact_F1={rec["contact_f1_vs_source"]:.3f}  '
                  f'Qcom={rec["q_com_path_l2"]:.2f}  wall={rec["wall_time_s"]:.1f}s')
        except Exception as e:
            traceback.print_exc()
            rec['status'] = 'error'
            rec['error'] = str(e)

        results['pairs'].append(rec)

    t_total = time.time() - t_all

    # ---- Aggregate metrics ----
    ok = [p for p in results['pairs'] if p.get('status') == 'ok']
    preds = [p['tgt_pred'] for p in ok if p.get('tgt_pred') is not None]
    pred_counts = {k: preds.count(k) for k in sorted(set(preds))}

    summary = {
        'n_authored': len(AUTHORED_PAIRS),
        'n_ok': len(ok),
        'mean_label_match': float(np.mean([int(p['label_match']) for p in ok])) if ok else 0.0,
        'mean_behavior_preserved': float(np.mean([int(p['behavior_preserved']) for p in ok])) if ok else 0.0,
        'mean_contact_f1_vs_source': float(np.mean([p['contact_f1_vs_source'] for p in ok])) if ok else None,
        'median_q_com_path_l2': float(np.median([p['q_com_path_l2'] for p in ok])) if ok else None,
        'median_q_heading_vel_l2': float(np.median([p['q_heading_vel_l2'] for p in ok])) if ok else None,
        'median_q_contact_sched_l2': float(np.median([p['q_contact_sched_l2'] for p in ok])) if ok else None,
        'median_q_cadence_abs_diff': float(np.median([p['q_cadence_abs_diff'] for p in ok])) if ok else None,
        'predicted_classes_distribution': pred_counts,
        'n_distinct_predicted_classes': len(pred_counts),
        'runtime_s': t_total,
    }

    # Go/No-Go verdict (Codex thresholds):
    #   absent label_match >= 0.25
    #   absent contact_f1_vs_source >= 0.35
    #   >= 3 distinct predicted classes
    verdict = {
        'threshold_label_match_0.25': summary['mean_label_match'] >= 0.25,
        'threshold_contact_f1_0.35': (summary['mean_contact_f1_vs_source'] or 0) >= 0.35,
        'threshold_diversity_3classes': summary['n_distinct_predicted_classes'] >= 3,
        'all_passed': False,
    }
    verdict['all_passed'] = all([
        verdict['threshold_label_match_0.25'],
        verdict['threshold_contact_f1_0.35'],
        verdict['threshold_diversity_3classes'],
    ])
    summary['codex_verdict'] = verdict

    # Reference: baseline numbers from prior runs (for paper-ready comparison)
    summary['baseline_context'] = {
        'prior_absent_label_match_ceiling_v2clf': 0.083,
        'prior_absent_contact_f1_K': 0.067,
        'prior_absent_contact_f1_H_v4': 0.459,
        'prior_absent_contact_f1_M2M_lite': 0.362,
    }

    results['summary'] = summary
    with open(OUT_DIR / 'metrics.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)

    print('\n======== SUMMARY ========')
    print(json.dumps(summary, indent=2, default=float))


if __name__ == '__main__':
    main()
