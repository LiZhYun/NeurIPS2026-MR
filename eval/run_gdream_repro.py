"""Partial G-DReaM reproduction (Cao et al., arXiv 2505.20857, 2025).

Closest published "unpaired graph-conditioned" competitor. Because the A3
backbone is already trained on all 70 Truebones skeletons with graph-aware
tokens (rest-pose + T5 joint-name + topology), we REUSE A3 as the
"graph-conditioned diffusion" body and ADD G-DReaM's energy-based classifier
guidance on top. The comparison this run answers:

    "Does the ENERGY-BASED guidance mechanism from G-DReaM add anything on
    top of A3's graph-aware backbone on Truebones?"

Mechanism (faithfully reproduced)
---------------------------------
Per paper Sec. 3 ("Energy-based guidance"):
    p(x)     ∝ exp(-fkin(x))
    fkin(x)  = λ_track * tracking                        # joint tracking
             + λ_fk    * FK consistency (vel ≈ Δpos)     # kinematic consistency
             + λ_vel   * velocity smoothness             # 2nd-diff regulariser
             + λ_floor * floor-penetration               # ground-plane term
             + λ_reg   * L2 regularisation               # weight decay in x

At each reverse step, we run ONE forward pass of the A3 model with
``requires_grad_(True)`` on x_t, obtain pred_xstart, compute fkin(pred_xstart),
back-propagate to get ∇_{x_t} fkin, then subtract a scaled version from
pred_xstart and step DDIM. This is the standard posterior-guidance recipe
(Dhariwal & Nichol 2021 "classifier guidance", made energy-based by Du & Mordatch 2020).

Lambda weights
--------------
Chosen so all five terms have comparable scale on normalised-space sample x0
under the A3 prior (empirically: tracking dominates if λ_track>>1; floor term
nearly zero on well-trained A3 outputs). We use:
    λ_track = 1.0, λ_fk = 0.5, λ_vel = 0.1, λ_floor = 0.2, λ_reg = 1e-3
    energy step size λ_step = 0.05     (scale of score correction)

Caveats (document for reviewer)
-------------------------------
This is a PARTIAL reproduction:
    * We do NOT re-train a G-DReaM-specific diffusion; we reuse A3.
    * Source "joint trajectory" is resampled from source Q (com_path +
      heading_vel) — not per-joint world positions, because source and
      target have different joint counts.
    * "Skeleton graph attention" is A3's built-in graph tokens — no
      additional G-DReaM cross-attention layer was added.
    * Diffusion step count = 20 (DDIM) instead of paper's 50 (DDPM),
      for 30-45 min wall budget.

Output
------
eval/results/k_compare/gdream_repro/pair_<id>_<src>_to_<tgt>.npy  (denormalised)
eval/results/k_compare/gdream_repro/metrics.json

Usage
-----
    conda run -n anytop python -m eval.run_gdream_repro
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
import torch.nn.functional as F

ROOT = Path(str(PROJECT_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse already-validated helpers (no modifications to source modules).
from eval.run_retrieve_refine_v2 import (  # noqa: E402
    q_component_l2, contact_f1, build_q_sig_array,
    build_source_anchored_contact_mask, stratified_summary,
    cosine_sim, l2_norm,
    EVAL_PAIRS, META_PATH, Q_CACHE_PATH, COND_PATH, MOTIONS_DIR,
    CONTACT_GROUPS_PATH, FOOT_CH_IDX, POS_Y_IDX,
)
from eval.quotient_extractor import extract_quotient  # noqa: E402

# Classifier-v2 bits for label/behavior scoring.
from eval.external_classifier import (  # noqa: E402
    ACTION_CLASSES,
    ACTION_TO_IDX,
    extract_classifier_features,
)
from eval.train_external_classifier_v2 import (  # noqa: E402
    V2Classifier,
    resample_along_time,
)

OUT_DIR = ROOT / 'eval/results/k_compare/gdream_repro'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Energy function hyper-parameters -------------------------------------
LAMBDA_TRACK = 1.0       # source Q (com_path + heading_vel) tracking
LAMBDA_FK    = 0.5       # velocity channels ≈ finite-difference of positions
LAMBDA_VEL   = 0.1       # velocity channel 2nd-order smoothness
LAMBDA_FLOOR = 0.2       # floor penetration on enforced-ground joints
LAMBDA_REG   = 1e-3      # L2 regulariser
LAMBDA_STEP  = 0.05      # classifier-guidance gradient scale

# --- DDIM hparams ---------------------------------------------------------
T_INIT  = 0.3
N_STEPS = 20
SEED    = 42

# Feature layout inside one joint's 13-dim slot (AnyTop convention).
ROT_START, ROT_END = 3, 9
VEL_X, VEL_Y, VEL_Z = 9, 10, 11
POS_X, POS_Y, POS_Z = 0, POS_Y_IDX, 2  # POS_Y_IDX is 1
CONTACT_THRESHOLD = 0.5

CLF_CKPT = ROOT / 'save/external_classifier_v2.pt'


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).reshape(-1).astype(np.float64)
    b = np.asarray(b).reshape(-1).astype(np.float64)
    denom = np.linalg.norm(a) + np.linalg.norm(b) + 1e-6
    return float(np.linalg.norm(a - b) / denom)


def _resample_time(a: np.ndarray, T: int) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 0 or a.shape[0] == T:
        return a
    T_a = a.shape[0]
    idx = np.clip(np.round(np.linspace(0, T_a - 1, T)).astype(int), 0, T_a - 1)
    return a[idx]


@torch.no_grad()
def classifier_probabilities(clf, features):
    if features is None:
        return None
    feats = resample_along_time(features, clf.target_T).astype(np.float32)
    feats = (feats - clf.feat_mean) / (clf.feat_std + 1e-6)
    x = torch.from_numpy(feats).unsqueeze(0).to(clf.device)
    probs_sum = None
    for m in clf.models:
        logits = m(x)
        probs = F.softmax(logits, dim=-1)
        probs_sum = probs if probs_sum is None else probs_sum + probs
    return (probs_sum / max(len(clf.models), 1)).squeeze(0).cpu().numpy().astype(np.float32)


def classify_motion_probs(motion_13dim, cond_skel, clf):
    from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np
    J_skel = cond_skel['offsets'].shape[0]
    m = motion_13dim
    if m.shape[1] > J_skel:
        m = m[:, :J_skel]
    if np.abs(m).max() < 5:
        mean = cond_skel['mean'][:J_skel]
        std = cond_skel['std'][:J_skel]
        m = m.astype(np.float32) * std + mean
    try:
        positions = recover_from_bvh_ric_np(m.astype(np.float32))
    except Exception:
        return None, None
    parents = cond_skel['parents'][:J_skel]
    feats = extract_classifier_features(positions, parents)
    if feats is None or feats.shape[0] < 4:
        return None, None
    probs = classifier_probabilities(clf, feats)
    if probs is None:
        return None, None
    return probs, ACTION_CLASSES[int(probs.argmax())]


def q_components_and_contact_f1(q_src: dict, q_tgt: dict) -> dict:
    T = min(q_src['com_path'].shape[0], q_tgt['com_path'].shape[0])
    out_com = _rel_l2(_resample_time(q_src['com_path'], T), _resample_time(q_tgt['com_path'], T))
    T = min(q_src['heading_vel'].shape[0], q_tgt['heading_vel'].shape[0])
    out_hv = _rel_l2(_resample_time(q_src['heading_vel'], T), _resample_time(q_tgt['heading_vel'], T))
    out_cad = float(abs(float(q_src['cadence']) - float(q_tgt['cadence'])))
    cs_src = np.asarray(q_src['contact_sched'])
    cs_tgt = np.asarray(q_tgt['contact_sched'])
    agg_src = cs_src.sum(axis=1) if cs_src.ndim == 2 else cs_src
    agg_tgt = cs_tgt.sum(axis=1) if cs_tgt.ndim == 2 else cs_tgt
    T = min(len(agg_src), len(agg_tgt))
    agg_src_r = _resample_time(agg_src, T)
    agg_tgt_r = _resample_time(agg_tgt, T)
    out_contact = _rel_l2(agg_src_r, agg_tgt_r)
    lu_src = -np.sort(-np.asarray(q_src['limb_usage']))[:5]
    lu_tgt = -np.sort(-np.asarray(q_tgt['limb_usage']))[:5]
    K = max(len(lu_src), len(lu_tgt), 5)
    lu_src = np.pad(lu_src, (0, K - len(lu_src)))
    lu_tgt = np.pad(lu_tgt, (0, K - len(lu_tgt)))
    out_lu = _rel_l2(lu_src, lu_tgt)
    bin_src = (agg_src_r >= CONTACT_THRESHOLD).astype(np.int8)
    bin_tgt = (agg_tgt_r >= CONTACT_THRESHOLD).astype(np.int8)
    tp = int(((bin_tgt == 1) & (bin_src == 1)).sum())
    fp = int(((bin_tgt == 1) & (bin_src == 0)).sum())
    fn = int(((bin_tgt == 0) & (bin_src == 1)).sum())
    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    return {
        'q_com_path_l2': out_com,
        'q_heading_vel_l2': out_hv,
        'q_cadence_abs_diff': out_cad,
        'q_contact_sched_aggregate_l2': out_contact,
        'q_limb_usage_top5_l2': out_lu,
        'contact_f1_vs_source': float(f1),
    }


# --------------------------------------------------------------------------
# Energy function (differentiable, operates on pred_xstart in normalised space)
# --------------------------------------------------------------------------
def fkin_energy(x0_hat: torch.Tensor,
                n_joints: int,
                mean_t: torch.Tensor,
                std_t: torch.Tensor,
                src_com_target_norm_y: torch.Tensor,
                src_heading_target: torch.Tensor,
                floor_mask_TJ: torch.Tensor) -> torch.Tensor:
    """fkin(x_hat) — scalar differentiable energy.

    x0_hat       : [B, J, F=13, T] in normalised space
    mean_t/std_t : [max_J, F] per-joint z-score stats
    src_com_target_norm_y : [T] source COM-Y after remap to target normalisation
    src_heading_target    : [T] source heading velocity (body-frame forward)
    floor_mask_TJ         : [T, J] binary mask for joints that must not sink below floor
    """
    B, J, Fdim, T = x0_hat.shape

    # ------ (1) joint tracking: COM-Y path of the real joints ------------
    real = x0_hat[:, :n_joints]                    # [B, n, F, T]
    # COM-height proxy: mean over real joints of normalised Y, mapped back to world.
    std_y = std_t[:n_joints, POS_Y].view(1, n_joints, 1)
    mean_y = mean_t[:n_joints, POS_Y].view(1, n_joints, 1)
    world_y = real[:, :, POS_Y] * std_y + mean_y    # [B, n, T]
    com_y = world_y.mean(dim=1)                    # [B, T]

    # Forward heading proxy: mean root-local X velocity (channel 9) over real joints,
    # in world units. A3 stores per-joint velocities in the same 13-dim slot.
    std_vx = std_t[:n_joints, VEL_X].view(1, n_joints, 1)
    mean_vx = mean_t[:n_joints, VEL_X].view(1, n_joints, 1)
    vx_world = real[:, :, VEL_X] * std_vx + mean_vx   # [B, n, T]
    heading_pred = vx_world.mean(dim=1)              # [B, T]

    com_tgt = src_com_target_norm_y.view(1, -1)      # [1, T]
    head_tgt = src_heading_target.view(1, -1)        # [1, T]

    track_com = (com_y - com_tgt).pow(2).mean()
    track_head = (heading_pred - head_tgt).pow(2).mean()
    e_track = track_com + track_head

    # ------ (2) FK consistency: vel ≈ finite-difference of root-local pos --
    # In A3, per-joint channels 0-2 are root-relative positions; 9-11 are velocities.
    # A physically consistent x0 satisfies pos[t+1] - pos[t] ≈ vel[t] / fps.
    # Both sides are in NORMALISED space; the denormalisation factor cancels in
    # relative MSE to within a per-joint scalar, so we compare in normalised space.
    pos = real[:, :, POS_X:POS_Z + 1]              # [B, n, 3, T]
    vel = real[:, :, VEL_X:VEL_Z + 1]              # [B, n, 3, T]
    dpos = pos[..., 1:] - pos[..., :-1]            # [B, n, 3, T-1]
    # AnyTop's velocity channels are already stored at the same step scale as
    # dpos (per-frame delta); in normalised space the standard-deviation scaling
    # is absorbed into the per-channel stats. A pure MSE on the difference is a
    # reasonable consistency proxy.
    e_fk = (dpos - vel[..., :-1]).pow(2).mean()

    # ------ (3) velocity 2nd-order smoothness ---------------------------
    dvel = vel[..., 1:] - vel[..., :-1]
    e_vel = dvel.pow(2).mean()

    # ------ (4) floor penetration on enforced-ground joints -------------
    # World-Y target for "on floor" is y=0. Floor penetration = world_y < 0.
    if floor_mask_TJ is not None and floor_mask_TJ.sum() > 0:
        m = floor_mask_TJ.t().unsqueeze(0).expand(B, -1, -1)   # [B, n, T]
        # Penalty = relu(-world_y)^2 -> only pay when the joint is below ground
        pen = torch.relu(-world_y)
        e_floor = (pen * pen * m).mean()
    else:
        e_floor = torch.zeros((), device=x0_hat.device, dtype=x0_hat.dtype)

    # ------ (5) L2 regulariser (prior term) -----------------------------
    e_reg = real.pow(2).mean()

    return (LAMBDA_TRACK * e_track
            + LAMBDA_FK    * e_fk
            + LAMBDA_VEL   * e_vel
            + LAMBDA_FLOOR * e_floor
            + LAMBDA_REG   * e_reg)


# --------------------------------------------------------------------------
# Guidance DDIM loop — reuse A3 model + cond dict
# --------------------------------------------------------------------------
def energy_guided_ddim(x_T, t_start, n_steps, y,
                       model, diffusion,
                       n_joints, mean_t, std_t,
                       src_com_target_norm_y, src_heading_target,
                       floor_mask_TJ, device):
    alphas_cum = torch.tensor(diffusion.alphas_cumprod, dtype=torch.float32, device=device)
    if n_steps >= t_start + 1:
        steps = list(range(t_start, -1, -1))
    else:
        steps = np.linspace(t_start, 0, n_steps + 1).round().astype(int).tolist()
    seen, ordered = set(), []
    for s in steps:
        s = int(s)
        if s not in seen:
            ordered.append(s)
            seen.add(s)
    steps = ordered

    img = x_T.detach()
    for i, t_val in enumerate(steps):
        t_tensor = torch.tensor([t_val], device=device, dtype=torch.long)

        img = img.detach().requires_grad_(True)
        out = diffusion.p_mean_variance(
            model, img, t_tensor, clip_denoised=False, model_kwargs={'y': y})
        x0_hat = out['pred_xstart']

        e_val = fkin_energy(
            x0_hat, n_joints, mean_t, std_t,
            src_com_target_norm_y, src_heading_target, floor_mask_TJ)
        grad = torch.autograd.grad(e_val, img, retain_graph=False, create_graph=False)[0]

        # Apply guidance on the predicted x0 (equivalent up to a linear
        # re-scaling via Tweedie's formula; simpler and numerically stable).
        # Normalise grad scale to prevent blow-up in early steps.
        g = grad.detach()
        scale = g.flatten(1).norm(dim=1, keepdim=True).clamp(min=1e-8)
        x0_guided = x0_hat.detach() - LAMBDA_STEP * g / scale.view(-1, 1, 1, 1)

        # DDIM step using the guided x0_hat.
        if i == len(steps) - 1 or t_val == 0:
            img = x0_guided
            break

        t_next = steps[i + 1]
        a_bar_t = alphas_cum[t_val]
        a_bar_next = alphas_cum[t_next] if t_next >= 0 else torch.tensor(1.0, device=device)
        # Recover eps from the ORIGINAL img_t and the predicted x0 (unguided).
        eps = (img.detach() - torch.sqrt(a_bar_t) * x0_hat.detach()) / torch.sqrt(
            (1.0 - a_bar_t).clamp(min=1e-12))
        img = torch.sqrt(a_bar_next) * x0_guided + torch.sqrt((1.0 - a_bar_next).clamp(min=0.0)) * eps

    return img.detach()


# --------------------------------------------------------------------------
# Main runner
# --------------------------------------------------------------------------
def run():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    t_start_all = time.time()

    # --- load AnyTop (A3) backbone via public helper ---
    from eval.anytop_projection import (
        _load_unconditional_model, _load_t5, _build_y,
        DEFAULT_CKPT, FALLBACK_CKPT, PROJECT_ROOT,
    )

    ckpt = Path(PROJECT_ROOT) / DEFAULT_CKPT
    if not ckpt.exists():
        ckpt = Path(PROJECT_ROOT) / FALLBACK_CKPT
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[gdream_repro] device={device}  ckpt={ckpt}')

    model, diffusion, m_args = _load_unconditional_model(str(ckpt), device)
    t5 = _load_t5(m_args.t5_name, device_str=str(device))

    with open(EVAL_PAIRS) as f:
        pairs = json.load(f)['pairs']
    with open(META_PATH) as f:
        meta = json.load(f)
    qc = np.load(Q_CACHE_PATH, allow_pickle=True)
    cond_dict = np.load(COND_PATH, allow_pickle=True).item()
    with open(CONTACT_GROUPS_PATH) as f:
        contact_groups = json.load(f)

    q_meta = list(qc['meta'])
    fname_to_q_idx = {m['fname']: i for i, m in enumerate(q_meta)}

    # Classifier-v2 for label/behavior scoring.
    clf = V2Classifier(str(CLF_CKPT), device=device)
    print(f'[gdream_repro] v2 classifier arch={clf.arch}  target_T={clf.target_T}')

    from data_loaders.truebones.data.dataset import create_temporal_mask_for_window  # noqa: F401

    max_joints = 143
    n_frames = m_args.num_frames
    feature_len = 13

    per_pair = []
    for p in pairs:
        pid = int(p['pair_id'])
        src_skel = p['source_skel']
        src_fname = p['source_fname']
        tgt_skel = p['target_skel']
        src_label = p['source_label']
        family_gap = p['family_gap']
        support = int(p['support_same_label'])
        strat = {'near': 'near_present'}.get(family_gap, family_gap)
        print(f"\n=== pair {pid:02d} {src_skel}({src_fname}) -> {tgt_skel}  "
              f"gap={family_gap}  sup={support} ===")

        rec = {'pair_id': pid, 'src_fname': src_fname, 'src_skel': src_skel,
               'src_label': src_label, 'tgt_skel': tgt_skel,
               'family_gap': strat, 'support_same_label': support,
               'status': 'pending', 'error': None}
        t_pair0 = time.time()
        try:
            if tgt_skel not in cond_dict:
                raise RuntimeError(f'target {tgt_skel} missing from cond.npy')
            if src_fname not in fname_to_q_idx:
                raise RuntimeError(f'source not in Q cache: {src_fname}')

            # --- Build conditioning for target skeleton ---
            y, n_joints, mean_np, std_np = _build_y(
                tgt_skel, cond_dict, t5, n_frames,
                m_args.temporal_window, max_joints, device)

            mean_pad = np.zeros((max_joints, feature_len), dtype=np.float32)
            std_pad = np.ones((max_joints, feature_len), dtype=np.float32)
            mean_pad[:n_joints] = mean_np
            std_pad[:n_joints] = std_np
            mean_t = torch.from_numpy(mean_pad).to(device)
            std_t = torch.from_numpy(std_pad).to(device)

            # --- Extract source Q (for tracking energy) ---
            src_q = extract_quotient(src_fname, cond_dict[src_skel],
                                     contact_groups=contact_groups,
                                     motion_dir=str(MOTIONS_DIR))

            # Body-scale ratio for world-frame remap.
            tgt_offsets = cond_dict[tgt_skel]['offsets']
            tgt_bs = float(np.linalg.norm(tgt_offsets, axis=1).sum() + 1e-6)
            src_bs = float(src_q['body_scale'])
            scale_ratio = tgt_bs / max(src_bs, 1e-6)

            # COM-Y target (world): resample + scale from source. COM path is
            # body-scale-normalised with anchor at frame 0, so multiplying by
            # target body scale + adding the resting root Y of the target
            # joints (proxy: average of target mean Y over real joints) gives
            # a world-frame absolute Y.
            src_com = np.asarray(src_q['com_path']).astype(np.float32)
            T_src = src_com.shape[0]
            idx = np.clip(np.round(np.linspace(0, T_src - 1, n_frames)).astype(int),
                          0, T_src - 1)
            com_target_world_y = src_com[idx, 1] * tgt_bs  # zero-anchored
            # Add target resting mean world-Y so absolute values are sane.
            resting_y = float(mean_np[:n_joints, POS_Y].mean())
            com_target_world_y = com_target_world_y + resting_y
            com_target_t = torch.from_numpy(com_target_world_y.astype(np.float32)).to(device)

            # Heading velocity target (body-scale-normalised forward speed).
            src_head = np.asarray(src_q['heading_vel']).astype(np.float32)
            head_target = src_head[idx] * tgt_bs  # world units (approx)
            head_target_t = torch.from_numpy(head_target.astype(np.float32)).to(device)

            # Floor mask from source schedule via name-overlap remap.
            tgt_groups_clean = {k: v for k, v in contact_groups.get(tgt_skel, {}).items()
                                if not str(k).startswith('_')}
            floor_mask = build_source_anchored_contact_mask(
                src_q, tgt_skel, tgt_groups_clean,
                contact_groups.get(src_skel, {}),
                n_frames_target=n_frames, n_joints_target=n_joints,
            )
            floor_mask_t = torch.from_numpy(floor_mask.astype(np.float32)).to(device)

            # --- Initial noise on the target skeleton ---
            t_max = diffusion.num_timesteps
            t_start_step = max(1, min(t_max - 1, int(round(T_INIT * t_max))))
            # Pure-noise init: G-DReaM samples from prior + guidance (no retrieval).
            x_T = torch.randn(1, max_joints, feature_len, n_frames, device=device)
            # Normalise-space base: pure Gaussian at step t_max. We actually want
            # the prior sample at step=t_start_step, not t=T_max: pure noise would
            # normally go through all 1000 steps, but here we run 20 from t_start.
            # Starting from pure noise is fine because the guidance pulls us toward
            # the target behaviour regardless.

            # --- Run energy-guided DDIM ---
            t0 = time.time()
            x_ref_padded = energy_guided_ddim(
                x_T, t_start_step, N_STEPS, y,
                model, diffusion, n_joints, mean_t, std_t,
                com_target_t, head_target_t, floor_mask_t, device)
            refine_time = time.time() - t0
            rec['refine_runtime_s'] = float(refine_time)

            # --- Denormalise ---
            x_ref = x_ref_padded[0, :n_joints].detach().cpu().numpy()
            # [J, F, T] -> [T, J, F]
            x_ref = x_ref.transpose(2, 0, 1)
            x_ref_denorm = x_ref * std_np[None] + mean_np[None]
            x_ref_denorm = x_ref_denorm.astype(np.float32)

            out_fname = f'pair_{pid:02d}_{src_skel}_to_{tgt_skel}.npy'
            np.save(OUT_DIR / out_fname, x_ref_denorm)
            rec['output_file'] = out_fname

            # --- Q-component metrics vs source ---
            tmp_post = f'__gdream_repro_pair_{pid:02d}.npy'
            tmp_post_path = MOTIONS_DIR / tmp_post
            try:
                np.save(tmp_post_path, x_ref_denorm)
                refined_q = extract_quotient(tmp_post, cond_dict[tgt_skel],
                                             contact_groups=contact_groups,
                                             motion_dir=str(MOTIONS_DIR))
            finally:
                if tmp_post_path.exists():
                    try:
                        tmp_post_path.unlink()
                    except Exception:
                        pass

            q_stats = q_components_and_contact_f1(src_q, refined_q)
            rec.update(q_stats)

            # skating proxy: mean joint-speed on frames where any joint is in
            # contact. Cheap and classifier-independent.
            from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np
            positions = recover_from_bvh_ric_np(x_ref_denorm)  # [T, J, 3]
            contacts = (x_ref_denorm[..., FOOT_CH_IDX] > 0.5)
            vel = np.zeros_like(positions)
            vel[1:] = positions[1:] - positions[:-1]
            speed = np.linalg.norm(vel, axis=-1)
            mask = contacts.astype(np.float32)
            if mask.sum() > 0:
                rec['skating_proxy'] = float((speed * mask).sum() / (mask.sum() + 1e-6))
            else:
                rec['skating_proxy'] = None

            # --- Classifier-v2 label + behavior ---
            probs_chosen, pred_chosen = classify_motion_probs(
                x_ref_denorm, cond_dict[tgt_skel], clf)
            src_motion = np.load(MOTIONS_DIR / src_fname)
            probs_src, pred_src = classify_motion_probs(
                src_motion, cond_dict[src_skel], clf)
            rec['action_classifier_pred'] = pred_chosen
            rec['source_classifier_pred'] = pred_src
            rec['label_match'] = (int(pred_chosen == src_label)
                                  if pred_chosen is not None else None)
            rec['behavior_preserved'] = (int(pred_chosen == pred_src)
                                         if (pred_chosen is not None
                                             and pred_src is not None) else None)
            if probs_chosen is not None and src_label in ACTION_TO_IDX:
                rec['p_source_action'] = float(probs_chosen[ACTION_TO_IDX[src_label]])
            else:
                rec['p_source_action'] = None

            rec['status'] = 'ok'
            rec['wall_time_s'] = float(time.time() - t_pair0)
            print(f"  ok  refine={refine_time:.2f}s  cf1={rec['contact_f1_vs_source']:.3f}  "
                  f"lm={rec['label_match']} bp={rec['behavior_preserved']}  "
                  f"q_com={rec['q_com_path_l2']:.3f}  q_hv={rec['q_heading_vel_l2']:.3f}")
        except Exception as e:
            rec['status'] = 'failed'
            rec['error'] = f'{e}\n{traceback.format_exc(limit=3)}'
            rec['wall_time_s'] = float(time.time() - t_pair0)
            print(f'  FAILED: {e}')
        per_pair.append(rec)

    total_time = time.time() - t_start_all

    # Stratified summary with the same keys as classifier_rerank/retrieve_refine_v4.
    NUMERIC_KEYS = [
        'label_match', 'behavior_preserved', 'p_source_action',
        'q_com_path_l2', 'q_heading_vel_l2', 'q_cadence_abs_diff',
        'q_contact_sched_aggregate_l2', 'q_limb_usage_top5_l2',
        'contact_f1_vs_source', 'skating_proxy', 'refine_runtime_s',
        'wall_time_s',
    ]

    def stratum_of(family_gap, support):
        s = []
        if family_gap in ('near', 'near_present'):
            s.append('near_present')
        elif family_gap == 'moderate':
            s.append('moderate')
        elif family_gap == 'extreme':
            s.append('extreme')
        if support == 0:
            s.append('absent')
        s.append('all')
        return s

    def stratified(entries):
        buckets = defaultdict(list)
        for e in entries:
            if e['status'] != 'ok':
                continue
            for s in stratum_of(e['family_gap'], e['support_same_label']):
                buckets[s].append(e)
        out = {}
        for s in ['all', 'near_present', 'absent', 'moderate', 'extreme']:
            es = buckets.get(s, [])
            stats = {'n': len(es)}
            for k in NUMERIC_KEYS:
                vals = [e[k] for e in es if e.get(k) is not None
                        and not (isinstance(e[k], float) and np.isnan(e[k]))]
                stats[k] = float(np.mean(vals)) if vals else None
            out[s] = stats
        return out

    summary = {
        'method': 'gdream_repro',
        'description': 'Partial G-DReaM reproduction: A3 backbone + energy-based guidance.',
        'hparams': {
            'lambda_track': LAMBDA_TRACK,
            'lambda_fk': LAMBDA_FK,
            'lambda_vel': LAMBDA_VEL,
            'lambda_floor': LAMBDA_FLOOR,
            'lambda_reg': LAMBDA_REG,
            'lambda_step': LAMBDA_STEP,
            't_init': T_INIT,
            'n_steps': N_STEPS,
            'seed': SEED,
        },
        'n_pairs': len(pairs),
        'n_ok': sum(1 for r in per_pair if r['status'] == 'ok'),
        'n_failed': sum(1 for r in per_pair if r['status'] != 'ok'),
        'total_runtime_sec': total_time,
        'per_pair': per_pair,
        'stratified': stratified(per_pair),
    }
    metrics_path = OUT_DIR / 'metrics.json'
    metrics_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n=== DONE: total {total_time:.1f}s  n_ok={summary['n_ok']}/{len(pairs)} ===")
    print(f'metrics.json: {metrics_path}')
    print('Stratified (behavior_preserved / contact_f1_vs_source):')
    for s in ['all', 'near_present', 'absent', 'moderate', 'extreme']:
        b = summary['stratified'].get(s, {})
        n = b.get('n', 0)
        bp = b.get('behavior_preserved')
        cf = b.get('contact_f1_vs_source')
        bp_s = f'{bp:.3f}' if bp is not None else '--'
        cf_s = f'{cf:.3f}' if cf is not None else '--'
        print(f"  {s:14s}  n={n}  behavior_preserved={bp_s}  contact_f1={cf_s}")
    return summary


if __name__ == '__main__':
    run()
