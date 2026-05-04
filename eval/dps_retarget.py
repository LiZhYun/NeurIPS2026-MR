"""Idea A — Diffusion Posterior Sampling (DPS) with behaviour-invariant measurement.

1-day falsification experiment. Goal: decide whether Idea A's signal is
dominated by its action-class logit term (reviewer prediction), by comparing
two measurement variants on 20 support-absent/moderate/extreme pairs:

  M_full        = [phase_trace, contact_pattern, psi_support, action_class_logit]
  M_no_action   = [phase_trace, contact_pattern, psi_support]

Implementation notes
--------------------
We operate on the AnyTop 13-dim feature tensor x ∈ R^{B, J, 13, T}.
* A3 predicts x_start (START_X mean type, cosine DDPM, 100 steps, predict x0).
* DPS guidance adds  −λ_m · ∇_{x_t} ‖M(x̂_0(x_t)) − M(x_src)‖²  per reverse step.
* We keep the measurement differentiable in the raw feature space:
    - phase_trace       : torch.fft.rfft of the contact channel summed across joints
    - contact_pattern   : time-mean of contact channel per depth-bin
    - psi_support       : time-mean + std of the height channel per depth-bin
    - action_class_logit: depth-binned per-frame classifier features → ActionClassifier
  Positions are approximated by concatenating the height channel with simple
  depth-binned statistics — NOT a full BVH recovery, so this is a proxy for M.
  For a falsification test that's adequate: if A collapses to the action logit
  under this proxy, it is even more likely to under the richer M.

We produce outputs in the motion feature space (denormalised, then world-frame
positions via recover_from_bvh_ric_np for downstream metrics).
"""
from __future__ import annotations
import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = str(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from os.path import join as pjoin
from typing import Optional, List, Dict

from eval.anytop_projection import (
    _load_unconditional_model, _load_t5, _build_y,
    POS_Y_IDX, FOOT_CH_IDX, ROT_START, ROT_END,
)
from eval.external_classifier import ActionClassifier, ACTION_CLASSES, ACTION_TO_IDX

# ---------------------------- measurement bits ----------------------------

def _depth_bins_from_parents(parents: np.ndarray, n_joints: int, n_bins: int = 8) -> np.ndarray:
    """Map each joint j ∈ [0, n_joints) to a depth-bin index in [0, n_bins)."""
    depth = np.zeros(n_joints, dtype=np.int64)
    for j in range(n_joints):
        d = 0
        cur = j
        while 0 <= parents[cur] != cur and d < n_joints:
            d += 1
            cur = int(parents[cur])
        depth[j] = d
    if depth.max() < 1:
        return np.zeros(n_joints, dtype=np.int64)
    depth_norm = depth / depth.max()
    return np.clip((depth_norm * n_bins).astype(np.int64), 0, n_bins - 1)


def _depth_bin_mean(feat: torch.Tensor, depth_bins: torch.Tensor,
                    n_bins: int) -> torch.Tensor:
    """Average feat[..., j, ...] across joints by depth-bin.

    feat: [B, J, T] — flat feature over joints and time
    depth_bins: [J] int64
    returns: [B, n_bins, T]
    """
    B, J, T = feat.shape
    out = feat.new_zeros(B, n_bins, T)
    cnt = feat.new_zeros(n_bins)
    for b in range(n_bins):
        mask = (depth_bins == b)
        c = int(mask.sum().item())
        if c == 0:
            continue
        out[:, b, :] = feat[:, mask, :].mean(dim=1)
        cnt[b] = c
    return out


def _phase_trace(contact_tj: torch.Tensor, T: int) -> torch.Tensor:
    """FFT magnitude spectrum of aggregate contact signal.

    contact_tj: [B, J, T] continuous contact values (differentiable)
    returns: [B, F]  F = T//2 + 1 (rfft bins)
    """
    s = contact_tj.mean(dim=1)                   # [B, T]
    s = s - s.mean(dim=-1, keepdim=True)         # zero-mean
    fft = torch.fft.rfft(s, dim=-1)              # [B, F]
    return torch.abs(fft)


def _contact_pattern(contact_tj: torch.Tensor, depth_bins: torch.Tensor,
                     n_bins: int) -> torch.Tensor:
    """Time-mean contact per depth-bin. [B, n_bins]"""
    dm = _depth_bin_mean(contact_tj, depth_bins, n_bins)       # [B, n_bins, T]
    return dm.mean(dim=-1)                                     # [B, n_bins]


def _psi_support(height_tj: torch.Tensor, depth_bins: torch.Tensor,
                 n_bins: int) -> torch.Tensor:
    """Depth-binned time-mean and time-std of the height channel.

    Returns [B, 2*n_bins]
    """
    dm = _depth_bin_mean(height_tj, depth_bins, n_bins)        # [B, n_bins, T]
    mu = dm.mean(dim=-1)
    sd = dm.std(dim=-1)
    return torch.cat([mu, sd], dim=-1)


def _clf_features_tj(pos_tj: torch.Tensor, vel_tj: torch.Tensor,
                     height_tj: torch.Tensor, depth_bins: torch.Tensor,
                     n_bins: int = 8) -> torch.Tensor:
    """Depth-binned per-frame features shaped [B, T-1, n_bins, 7] for classifier.

    Mirrors extract_classifier_features but operates differentiably on the
    feature-tensor proxies (height, velocity, centroid_dist approximated as
    |root-relative position|).

    pos_tj: [B, J, 3, T], vel_tj: [B, J, 3, T], height_tj: [B, J, T]
    """
    B, J, _, T = vel_tj.shape
    pos_rel_mag = torch.sqrt((pos_tj ** 2).sum(dim=2) + 1e-8)   # [B, J, T]
    vel_mag = torch.sqrt((vel_tj ** 2).sum(dim=2) + 1e-8)        # [B, J, T]
    height_rel = height_tj - height_tj.min(dim=1, keepdim=True)[0]
    centroid_speed = vel_mag.mean(dim=1, keepdim=True)           # [B, 1, T]

    # bin-wise aggregations
    T_minus_1 = T - 1

    out = vel_tj.new_zeros(B, T_minus_1, n_bins, 7)
    for b in range(n_bins):
        mask = (depth_bins == b)
        c = int(mask.sum().item())
        if c == 0:
            continue
        vm = vel_mag[:, mask, 1:]           # [B, Cb, T-1]
        hr = height_rel[:, mask, 1:]
        pm = pos_rel_mag[:, mask, 1:]
        out[:, :, b, 0] = vm.mean(dim=1)
        out[:, :, b, 1] = vm.max(dim=1).values
        out[:, :, b, 2] = hr.mean(dim=1)
        out[:, :, b, 3] = pm.mean(dim=1)
        out[:, :, b, 4] = pm.max(dim=1).values
        out[:, :, b, 5] = float(c) / max(J, 1)
        out[:, :, b, 6] = centroid_speed.squeeze(1)[:, 1:]
    return out


def measurement(x: torch.Tensor, mean_t: torch.Tensor, std_t: torch.Tensor,
                depth_bins: torch.Tensor, n_joints: int, n_bins: int,
                include_action: bool,
                classifier: Optional[ActionClassifier],
                action_idx: int) -> Dict[str, torch.Tensor]:
    """Compute M(x) on a normalised motion tensor [B, J, 13, T].

    Returns a dict with tensors:
      phase_trace      [B, F]
      contact_pattern  [B, n_bins]
      psi_support      [B, 2*n_bins]
      action_logit     [B]    only if include_action=True
    """
    B, J, C, T = x.shape
    assert C == 13
    # Denormalise only channels we read
    mean = mean_t.view(1, J, C, 1)        # [1, J, 13, 1]
    std = std_t.view(1, J, C, 1)
    x_de = x * std + mean                 # [B, J, 13, T]

    # Slice real joints only (depth_bins length)
    x_de = x_de[:, :n_joints, :, :]

    contact_tj = torch.sigmoid(2.0 * x_de[:, :, FOOT_CH_IDX, :])   # [B, J', T] squash to (0,1)
    height_tj = x_de[:, :, POS_Y_IDX, :]                           # [B, J', T]
    # Position proxy = concatenation of raw position channels
    pos_tj = x_de[:, :, :3, :]                                     # [B, J', 3, T]
    vel_tj = x_de[:, :, 9:12, :]                                   # [B, J', 3, T]

    out: Dict[str, torch.Tensor] = {}
    out['phase_trace'] = _phase_trace(contact_tj, T)
    out['contact_pattern'] = _contact_pattern(contact_tj, depth_bins, n_bins)
    out['psi_support'] = _psi_support(height_tj, depth_bins, n_bins)
    if include_action and classifier is not None:
        feats = _clf_features_tj(pos_tj, vel_tj, height_tj, depth_bins, n_bins)
        logits = classifier(feats)                                # [B, 12]
        out['action_logit'] = logits[:, action_idx]
        out['action_logits_all'] = logits
        out['action_idx'] = torch.tensor(action_idx, device=x.device)
    return out


def measurement_diff_sq(m_pred: Dict[str, torch.Tensor],
                        m_src: Dict[str, torch.Tensor],
                        include_action: bool,
                        scales: Optional[Dict[str, float]] = None) -> torch.Tensor:
    """Squared-error loss between measurement dicts, scale-normalised.

    Each component is divided by (‖m_src[key]‖² + eps) so weights are comparable.
    When include_action=True, we include an action cross-entropy-style term:
      −log softmax(action_logit_vec)[src_label]  proxied by (−action_logit)
    via the source-class scalar logit we already pass in.
    """
    eps = 1e-6
    loss = 0.0
    keys = ['phase_trace', 'contact_pattern', 'psi_support']
    for k in keys:
        a = m_pred[k]
        b = m_src[k]
        denom = (b.pow(2).mean() + eps)
        loss = loss + ((a - b) ** 2).mean() / denom
    if include_action:
        # Cross-entropy style: minimise −log softmax(logits)[src_idx]
        logits_all = m_pred['action_logits_all']                  # [B, 12]
        src_idx = int(m_pred['action_idx'].item())
        log_prob = F.log_softmax(logits_all, dim=-1)[:, src_idx]
        loss = loss + (-log_prob).mean()
    return loss


# ---------------------------- DPS inference -------------------------------

def _load_classifier(device: torch.device) -> ActionClassifier:
    ckpt = pjoin(PROJECT_ROOT, 'save/external_classifier.pt')
    state = torch.load(ckpt, map_location='cpu', weights_only=False)
    clf = ActionClassifier()
    clf.load_state_dict(state['model'])
    clf.to(device).eval()
    for p in clf.parameters():
        p.requires_grad_(False)
    return clf


def _compute_source_measurement(source_motion: np.ndarray,
                                source_mean: np.ndarray, source_std: np.ndarray,
                                source_parents: np.ndarray, source_n_joints: int,
                                n_bins: int, include_action: bool,
                                classifier: Optional[ActionClassifier],
                                action_idx: int,
                                max_joints: int, n_frames: int,
                                device: torch.device) -> Dict[str, torch.Tensor]:
    """Compute M on the NORMALISED source motion in the same feature frame as
    the target generation.

    We normalise source by ITS OWN stats so features live in 'de-normalised' world
    space inside `measurement()`. Padded to (max_joints, 13, n_frames)."""
    T_real = source_motion.shape[0]
    T = min(T_real, n_frames)
    src = source_motion[:T].astype(np.float32)                   # [T, J, 13]
    # Normalise for measurement (we pass source_mean/std to un-normalise inside M).
    norm = np.nan_to_num((src - source_mean[None]) / (source_std[None] + 1e-6))
    pad = np.zeros((max_joints, 13, n_frames), dtype=np.float32)
    pad[:source_n_joints, :, :T] = norm.transpose(1, 2, 0)
    x = torch.from_numpy(pad).unsqueeze(0).to(device)             # [1, J, 13, T]

    mean_pad = np.zeros((max_joints, 13), dtype=np.float32)
    std_pad = np.ones((max_joints, 13), dtype=np.float32)
    mean_pad[:source_n_joints] = source_mean
    std_pad[:source_n_joints] = source_std + 1e-6
    mean_t = torch.from_numpy(mean_pad).to(device)
    std_t = torch.from_numpy(std_pad).to(device)

    depth_bins_np = _depth_bins_from_parents(source_parents, source_n_joints, n_bins)
    depth_bins = torch.from_numpy(depth_bins_np).to(device)

    with torch.no_grad():
        m = measurement(x, mean_t, std_t, depth_bins, source_n_joints, n_bins,
                        include_action, classifier, action_idx)
    return m


def dps_retarget(source_motion: np.ndarray,
                 source_skel: str,
                 source_parents: np.ndarray,
                 source_mean: np.ndarray,
                 source_std: np.ndarray,
                 source_label: str,
                 target_skel: str,
                 cond_dict: dict,
                 t5,
                 model,
                 diffusion,
                 m_args,
                 classifier: Optional[ActionClassifier],
                 M_variant: str = 'full',
                 n_steps: int = 50,
                 lambda_m: float = 0.1,
                 n_bins: int = 8,
                 seed: int = 42,
                 device: str = 'cuda') -> Dict:
    """DPS retarget: generate a motion on target_skel whose measurement matches
    that of source_motion.

    M_variant: 'full' or 'no_action'
    """
    dev = torch.device(device)
    include_action = (M_variant == 'full')

    max_joints = 143
    feature_len = 13
    n_frames = m_args.num_frames

    # --- Target cond
    y, n_joints_tgt, mean_np, std_np = _build_y(
        target_skel, cond_dict, t5, n_frames,
        m_args.temporal_window, max_joints, dev)
    mean_pad = np.zeros((max_joints, feature_len), dtype=np.float32)
    std_pad = np.ones((max_joints, feature_len), dtype=np.float32)
    mean_pad[:n_joints_tgt] = mean_np
    std_pad[:n_joints_tgt] = std_np + 1e-6
    mean_t = torch.from_numpy(mean_pad).to(dev)
    std_t = torch.from_numpy(std_pad).to(dev)

    # Depth-bins on TARGET skeleton
    tgt_parents = np.array(cond_dict[target_skel]['parents'][:n_joints_tgt], dtype=np.int64)
    depth_bins_tgt_np = _depth_bins_from_parents(tgt_parents, n_joints_tgt, n_bins)
    depth_bins_tgt = torch.from_numpy(depth_bins_tgt_np).to(dev)

    # --- Source measurement (fixed target in feature space)
    action_idx = ACTION_TO_IDX.get(source_label, ACTION_TO_IDX['other'])
    m_src = _compute_source_measurement(
        source_motion, source_mean, source_std,
        np.array(source_parents, dtype=np.int64),
        int(source_motion.shape[1]),
        n_bins, include_action, classifier, action_idx,
        max_joints, n_frames, dev)
    # Detach source from any graph
    m_src = {k: v.detach() for k, v in m_src.items()}

    # --- Initialise x_T ~ N(0, I) in normalised target space
    torch.manual_seed(seed)
    x_t = torch.randn(1, max_joints, feature_len, n_frames, device=dev)

    t_max = diffusion.num_timesteps  # 100
    steps = list(range(t_max - 1, -1, -1))
    if n_steps < t_max:
        idx = np.linspace(0, t_max - 1, n_steps).round().astype(int)[::-1]
        steps = idx.tolist()
    alphas_cum = torch.tensor(diffusion.alphas_cumprod, dtype=torch.float32, device=dev)

    tic = time.time()
    torch.cuda.reset_peak_memory_stats(dev)

    for i, t_val in enumerate(steps):
        t_tensor = torch.tensor([t_val], device=dev, dtype=torch.long)

        x_t.requires_grad_(True)
        # Forward: need grad w.r.t. x_t → cannot wrap in no_grad, but the model
        # params are frozen so activations live only where needed.
        model_out = model(x_t, diffusion._scale_timesteps(t_tensor),
                          **{'y': y})
        # A3 predicts x_start directly
        x0_hat = model_out

        # Compute measurement on predicted x0 (normalised space)
        m_pred = measurement(x0_hat, mean_t, std_t, depth_bins_tgt,
                             n_joints_tgt, n_bins, include_action,
                             classifier, action_idx)
        loss = measurement_diff_sq(m_pred, m_src, include_action)
        grad = torch.autograd.grad(loss, x_t)[0].detach()
        # DPS paper (Chung et al. 2023) normalises step by sqrt(loss) so step size
        # is measurement-scale independent. grad ← grad / sqrt(loss + eps).
        grad = grad / torch.sqrt(loss.detach() + 1e-8)
        # Clip gradient norm for stability
        g_norm = grad.norm()
        max_norm = 50.0
        if g_norm > max_norm:
            grad = grad * (max_norm / g_norm)
        x_t = x_t.detach()
        x0_hat = x0_hat.detach()

        # DDIM step (η=0): eps = (x_t - sqrt(α_bar_t) x̂_0) / sqrt(1−α_bar_t)
        a_bar_t = alphas_cum[t_val]
        if i == len(steps) - 1:
            # last step — jump to x̂_0 and apply DPS correction once more
            x_next = x0_hat - lambda_m * grad
            x_t = x_next
            break
        t_next = steps[i + 1]
        a_bar_next = alphas_cum[t_next] if t_next >= 0 else torch.tensor(
            1.0, device=dev)
        eps = (x_t - torch.sqrt(a_bar_t) * x0_hat) / torch.sqrt(
            (1.0 - a_bar_t).clamp(min=1e-12))
        x_mean = torch.sqrt(a_bar_next) * x0_hat + torch.sqrt(
            (1.0 - a_bar_next).clamp(min=0.0)) * eps
        # DPS guidance — gradient wrt x_t, applied after DDIM mean
        x_t = x_mean - lambda_m * grad

    runtime = time.time() - tic
    peak_mem_bytes = torch.cuda.max_memory_allocated(dev)

    # --- Denormalise and crop
    x_final = x_t.detach()
    x_ref = x_final[0, :n_joints_tgt].cpu().numpy()            # [J, 13, T]
    x_ref = x_ref.transpose(2, 0, 1)                           # [T, J, 13]
    x_ref_denorm = x_ref * std_np[None] + mean_np[None]

    return {
        'x_norm': x_ref,                 # normalised target-space features
        'x_denorm': x_ref_denorm.astype(np.float32),
        'runtime_s': float(runtime),
        'gpu_peak_mb': float(peak_mem_bytes / 1024 / 1024),
        'n_joints_tgt': int(n_joints_tgt),
        'target_mean': mean_np.astype(np.float32),
        'target_std': std_np.astype(np.float32),
    }


# ------------------------------- metrics ----------------------------------

def _positions_from_denorm(x_denorm: np.ndarray) -> np.ndarray:
    """Recover world-frame positions [T, J, 3] from denormalised features."""
    from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np
    return recover_from_bvh_ric_np(x_denorm.astype(np.float32))


def _classify_positions(positions: np.ndarray, parents: np.ndarray,
                        classifier: ActionClassifier, device) -> np.ndarray:
    from eval.external_classifier import extract_classifier_features
    feats = extract_classifier_features(positions, parents)
    if feats is None:
        return np.zeros(12, dtype=np.float32)
    T = feats.shape[0]
    if T < 64:
        feats = np.pad(feats, ((0, 64 - T), (0, 0), (0, 0)))
    else:
        idx = np.linspace(0, T - 1, 64).astype(int)
        feats = feats[idx]
    with torch.no_grad():
        x = torch.from_numpy(feats[None]).float().to(device)
        logits = classifier(x)[0].cpu().numpy()
    return logits


def _contact_fft_dominant(contact_schedule_T: np.ndarray) -> float:
    """Dominant-frequency bin of a contact schedule."""
    s = contact_schedule_T.astype(np.float32)
    s = s - s.mean()
    if s.std() < 1e-8:
        return 0.0
    fft = np.fft.rfft(s)
    mag = np.abs(fft)
    # Exclude DC
    if mag.shape[0] < 2:
        return 0.0
    idx = int(np.argmax(mag[1:]) + 1)
    return float(idx)


def compute_metrics(x_denorm_gen: np.ndarray, pair: dict, cond_dict: dict,
                     target_mean_list: List[np.ndarray],
                     classifier: ActionClassifier,
                     source_motion: np.ndarray, source_parents: np.ndarray,
                     contact_groups: Optional[dict], device) -> dict:
    """Compute per-pair metrics."""
    tgt_skel = pair['target_skel']
    src_label = pair['source_label']
    info = cond_dict[tgt_skel]
    n_j_tgt = len(info['joints_names'])
    tgt_parents = np.array(info['parents'][:n_j_tgt], dtype=np.int64)

    # Recover world positions
    try:
        positions_gen = _positions_from_denorm(x_denorm_gen)
    except Exception as e:
        positions_gen = None
    # Action accuracy
    action_logits = _classify_positions(positions_gen, tgt_parents, classifier,
                                        device) if positions_gen is not None else np.zeros(12, dtype=np.float32)
    pred_class = int(np.argmax(action_logits))
    src_class_idx = ACTION_TO_IDX.get(src_label, ACTION_TO_IDX['other'])
    action_correct = int(pred_class == src_class_idx)
    # Probability of source class
    probs = np.exp(action_logits - action_logits.max())
    probs /= (probs.sum() + 1e-9)
    action_prob_src = float(probs[src_class_idx])

    # Content distance from target prior mean (L2 on normalised features, proxy for
    # "snap back to target prior"). Smaller ⇒ more collapsed to prior.
    if target_mean_list is not None and len(target_mean_list) > 0:
        xg = x_denorm_gen[:, :n_j_tgt, :]
        clips_ok = [m[:xg.shape[0], :n_j_tgt, :]
                    for m in target_mean_list
                    if m.shape[1] >= n_j_tgt and m.shape[0] >= xg.shape[0]]
        if not clips_ok:
            # Fallback: crop+zero-pad each available clip
            clips_ok = []
            for m in target_mean_list:
                m_ = m[:, :n_j_tgt, :] if m.shape[1] >= n_j_tgt else None
                if m_ is None: continue
                if m_.shape[0] < xg.shape[0]:
                    pad = np.zeros((xg.shape[0] - m_.shape[0], n_j_tgt, 13), dtype=np.float32)
                    m_ = np.concatenate([m_, pad], axis=0)
                else:
                    m_ = m_[:xg.shape[0]]
                clips_ok.append(m_)
        if not clips_ok:
            content_dist = float('nan')
        else:
            mu_mean = np.stack(clips_ok, axis=0).mean(axis=0)
            content_dist = float(np.sqrt(((xg - mu_mean) ** 2).mean()))
    else:
        content_dist = float('nan')

    # Phase preservation: dominant frequency of contact schedule (aggregated)
    contact_gen = x_denorm_gen[:, :n_j_tgt, FOOT_CH_IDX]           # [T, J]
    contact_gen_b = (contact_gen > 0.5).astype(np.float32).mean(axis=1)
    contact_src = source_motion[:, :, FOOT_CH_IDX]
    contact_src_b = (contact_src > 0.5).astype(np.float32).mean(axis=1)
    f_gen = _contact_fft_dominant(contact_gen_b)
    f_src = _contact_fft_dominant(contact_src_b)
    phase_dist = float(abs(f_gen - f_src))

    # Contact-schedule MAE via target contact_groups
    if contact_groups is not None and tgt_skel in contact_groups:
        groups = contact_groups[tgt_skel]
        names = [k for k in sorted(groups.keys()) if not k.startswith('_')]
        frac_gen = {}
        for name in names:
            idxs = [j for j in groups[name] if 0 <= j < n_j_tgt]
            if idxs:
                frac_gen[name] = float((contact_gen[:, idxs] > 0.5).mean())
        # Source groups (by source skel)
        src_skel = pair['source_skel']
        mae_val = float('nan')
        if src_skel in contact_groups:
            src_groups = contact_groups[src_skel]
            src_names = [k for k in sorted(src_groups.keys()) if not k.startswith('_')]
            # Match by name overlap
            common = set(frac_gen.keys()) & set(src_names)
            if common:
                frac_src = {}
                for name in common:
                    idxs = [j for j in src_groups[name] if 0 <= j < source_motion.shape[1]]
                    if idxs:
                        frac_src[name] = float((contact_src[:, idxs] > 0.5).mean())
                diffs = [abs(frac_gen[n] - frac_src[n]) for n in common if n in frac_src]
                mae_val = float(np.mean(diffs)) if diffs else float('nan')
        contact_mae = mae_val
    else:
        contact_mae = float('nan')

    return {
        'action_correct': action_correct,
        'action_prob_src': action_prob_src,
        'content_dist': content_dist,
        'phase_dist': phase_dist,
        'contact_mae': contact_mae,
    }


# ------------------------------ driver ------------------------------------

def _load_contact_groups():
    path = pjoin(PROJECT_ROOT, 'eval/quotient_assets/contact_groups.json')
    with open(path) as f:
        return json.load(f)


def _load_eval_pairs(filter_gaps=('absent', 'moderate', 'extreme')):
    with open(pjoin(PROJECT_ROOT, 'idea-stage/eval_pairs.json')) as f:
        data = json.load(f)
    pairs = [p for p in data['pairs'] if p['family_gap'] in filter_gaps]
    pairs.sort(key=lambda p: p['pair_id'])
    return pairs


def _load_target_clips(target_skel: str, motion_dir: str, n_max: int = 6) -> List[np.ndarray]:
    out = []
    for f in sorted(os.listdir(motion_dir)):
        if f.startswith(f'{target_skel}_'):
            try:
                out.append(np.load(pjoin(motion_dir, f)).astype(np.float32))
            except Exception:
                continue
            if len(out) >= n_max:
                break
    return out


def run_variant(variant: str, ckpt_path: str, device: str, lambda_m: float,
                n_steps: int, max_pairs: Optional[int] = None):
    from data_loaders.truebones.truebones_utils.get_opt import get_opt
    from data_loaders.truebones.truebones_utils.param_utils import DATASET_DIR

    dev = torch.device(device)
    model, diffusion, m_args = _load_unconditional_model(ckpt_path, dev)
    t5 = _load_t5(m_args.t5_name, device_str=device)
    classifier = _load_classifier(dev)

    opt = get_opt(0)
    cond_dict = np.load(opt.cond_file, allow_pickle=True).item()
    motion_dir = opt.motion_dir
    contact_groups = _load_contact_groups()

    pairs = _load_eval_pairs()
    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    out_dir = pjoin(PROJECT_ROOT, f'eval/results/k_compare/A_{variant}')
    os.makedirs(out_dir, exist_ok=True)

    include_action = (variant == 'full')

    all_metrics = []
    for pair in pairs:
        src_fname = pair['source_fname']
        src_skel = pair['source_skel']
        tgt_skel = pair['target_skel']
        if tgt_skel not in cond_dict or src_skel not in cond_dict:
            print(f"[skip] pair {pair['pair_id']} missing cond")
            continue
        src_info = cond_dict[src_skel]
        src_n_j = len(src_info['joints_names'])
        src_parents = np.array(src_info['parents'][:src_n_j], dtype=np.int64)
        src_mean = src_info['mean']
        src_std = src_info['std']
        try:
            src_raw = np.load(pjoin(motion_dir, src_fname)).astype(np.float32)
        except Exception as e:
            print(f"[skip] can't load {src_fname}: {e}")
            continue
        # Denormalise source for measurement: _compute_source_measurement expects
        # the source in de-normalised "data" form (we re-normalise inside).
        src_denorm = src_raw[:, :src_n_j] * (src_std + 1e-6) + src_mean   # [T, J_src, 13]

        print(f"\n[pair {pair['pair_id']}] {src_skel}({pair['source_label']}) → {tgt_skel}  [{pair['family_gap']}]")
        try:
            res = dps_retarget(
                source_motion=src_denorm,
                source_skel=src_skel,
                source_parents=src_parents,
                source_mean=src_mean,
                source_std=src_std,
                source_label=pair['source_label'],
                target_skel=tgt_skel,
                cond_dict=cond_dict,
                t5=t5, model=model, diffusion=diffusion, m_args=m_args,
                classifier=classifier,
                M_variant=('full' if include_action else 'no_action'),
                n_steps=n_steps, lambda_m=lambda_m, device=device)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  ERROR: {e}")
            continue

        out_fn = pjoin(out_dir, f"pair_{pair['pair_id']:02d}_{src_skel}_to_{tgt_skel}.npy")
        np.save(out_fn, res['x_denorm'])

        # Target prior clips for content distance
        tgt_clips = _load_target_clips(tgt_skel, motion_dir, n_max=6)
        # Denormalise target clips for consistent scale
        tgt_info = cond_dict[tgt_skel]
        tgt_mean = tgt_info['mean']
        tgt_std = tgt_info['std']
        tgt_nj = len(tgt_info['joints_names'])
        tgt_clips_d = []
        for c in tgt_clips:
            try:
                c_trim = c[:, :tgt_nj] * (tgt_std + 1e-6) + tgt_mean
                tgt_clips_d.append(c_trim)
            except Exception:
                continue

        m = compute_metrics(res['x_denorm'], pair, cond_dict, tgt_clips_d,
                            classifier, src_denorm, src_parents, contact_groups, dev)
        m.update({
            'pair_id': int(pair['pair_id']),
            'family_gap': pair['family_gap'],
            'source_label': pair['source_label'],
            'source_skel': src_skel,
            'target_skel': tgt_skel,
            'runtime_s': float(res['runtime_s']),
            'gpu_peak_mb': float(res['gpu_peak_mb']),
            'out_path': out_fn,
        })
        all_metrics.append(m)
        print(f"  action_correct={m['action_correct']}  prob_src={m['action_prob_src']:.3f}  "
              f"content={m['content_dist']:.3f}  phase={m['phase_dist']:.2f}  "
              f"contact_mae={m['contact_mae']:.3f}  t={m['runtime_s']:.1f}s  mem={m['gpu_peak_mb']:.0f}MB")

    # Aggregate
    def _mean(xs):
        xs = [x for x in xs if isinstance(x, (int, float)) and not np.isnan(x)]
        return float(np.mean(xs)) if xs else float('nan')
    summary = {
        'variant': variant,
        'n_pairs': len(all_metrics),
        'action_accuracy': _mean([m['action_correct'] for m in all_metrics]),
        'action_prob_src_mean': _mean([m['action_prob_src'] for m in all_metrics]),
        'content_dist_mean': _mean([m['content_dist'] for m in all_metrics]),
        'phase_dist_mean': _mean([m['phase_dist'] for m in all_metrics]),
        'contact_mae_mean': _mean([m['contact_mae'] for m in all_metrics]),
        'wall_time_mean_s': _mean([m['runtime_s'] for m in all_metrics]),
        'wall_time_total_s': float(sum(m['runtime_s'] for m in all_metrics)),
        'gpu_peak_mb_max': float(max((m['gpu_peak_mb'] for m in all_metrics), default=0)),
    }
    # Stratified by family_gap
    for gap in ('absent', 'moderate', 'extreme'):
        grp = [m for m in all_metrics if m['family_gap'] == gap]
        summary[f'{gap}_n'] = len(grp)
        summary[f'{gap}_action_acc'] = _mean([m['action_correct'] for m in grp])
        summary[f'{gap}_content_dist'] = _mean([m['content_dist'] for m in grp])
        summary[f'{gap}_phase_dist'] = _mean([m['phase_dist'] for m in grp])

    # Stratified by support_same_label (support-absent = sup==0)
    for m in all_metrics:
        m['support_same_label'] = next(
            (p['support_same_label'] for p in _load_eval_pairs()
             if p['pair_id'] == m['pair_id']), -1)
    sup_absent = [m for m in all_metrics if m.get('support_same_label', -1) == 0]
    summary['sup_absent_n'] = len(sup_absent)
    summary['sup_absent_action_acc'] = _mean([m['action_correct'] for m in sup_absent])
    summary['sup_absent_content_dist'] = _mean([m['content_dist'] for m in sup_absent])
    summary['sup_absent_phase_dist'] = _mean([m['phase_dist'] for m in sup_absent])

    out_json = pjoin(out_dir, 'metrics.json')
    with open(out_json, 'w') as f:
        json.dump({'summary': summary, 'per_pair': all_metrics}, f, indent=2)
    print(f"\n[{variant}] SUMMARY {summary}")
    print(f"[{variant}] wrote {out_json}")
    return summary, all_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default='save/A3_baseline_dataset_truebones_bs_2_latentdim_256/model000229999.pt')
    p.add_argument('--device', default='cuda')
    p.add_argument('--n_steps', type=int, default=50)
    p.add_argument('--lambda_m', type=float, default=0.1)
    p.add_argument('--variant', choices=['full', 'no_action', 'both'], default='both')
    p.add_argument('--max_pairs', type=int, default=None)
    args = p.parse_args()

    ckpt_path = args.ckpt
    if not os.path.isabs(ckpt_path):
        ckpt_path = pjoin(PROJECT_ROOT, ckpt_path)

    variants = ['full', 'no_action'] if args.variant == 'both' else [args.variant]
    all_summaries = {}
    for v in variants:
        s, _ = run_variant(v, ckpt_path, args.device, args.lambda_m, args.n_steps,
                           args.max_pairs)
        all_summaries[v] = s

    # Falsification verdict
    if 'full' in all_summaries and 'no_action' in all_summaries:
        a_full = all_summaries['full']['action_accuracy']
        a_none = all_summaries['no_action']['action_accuracy']
        chance = 1.0 / 12
        # Signal attributable to action term = (a_full - a_none) / (a_full - chance)
        denom = max(a_full - chance, 1e-3)
        signal_frac = (a_full - a_none) / denom
        verdict = {
            'action_acc_full': a_full,
            'action_acc_no_action': a_none,
            'chance_level': chance,
            'action_term_signal_fraction': float(signal_frac),
            'snap_back_confirmed': bool(signal_frac >= 0.8 and a_none <= 0.15),
            'non_trivial_signal_without_action': bool(a_none >= 0.30),
        }
        with open(pjoin(PROJECT_ROOT, 'eval/results/k_compare/verdict.json'), 'w') as f:
            json.dump(verdict, f, indent=2)
        print("\n=== FALSIFICATION VERDICT ===")
        print(json.dumps(verdict, indent=2))


if __name__ == '__main__':
    main()
