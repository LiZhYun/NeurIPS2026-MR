"""Pilot C (verified MotionClone) — 30-pair run on A3.

Reference: arXiv 2406.05338 (ICLR 2025), verification notes in
`idea-stage/LIT_REVIEW_VERIFICATION_2026_04_15.md`.

Verified mechanism (NOT the earlier "gradient-injection" version):
  - Sparse temporal-attention weights from the frozen video-diffusion prior.
  - Single-step extraction (one fixed timestep, not every timestep).
  - Location-aware semantic guidance (loss supervised only at non-zero
    positions of the sparse source map).
  - Training-free.

Pipeline
--------
1.  Load A3 checkpoint. Install `TemporalAttnTap` on all 4 decoder layers
    (runtime monkey-patch — no file edits).
2.  Source pass (on source skeleton, source motion):
      x_s = source motion (normalised, padded to max_joints/n_frames).
      t_star = round(0.3 * diffusion.num_timesteps).
      x_s_noisy = q_sample(x_s, t_star).
      Run A3 forward on (x_s_noisy, y_src). Collect per-layer
      `[B*J_src, H, T, T]` attention maps.
      Aggregate across joints: mean over J_src → `[B, H, T, T]`.
      Sparsify: top-K per query row (K = 5).
3.  Target generation on target skeleton:
      x_T ~ N(0, I) on target (padded).
      DDIM reverse 50 steps with eta = 0.
      At the step whose `t_val` is closest to t_star (single step only):
        - run a grad-enabled forward to compute target temporal-attn maps
          (same aggregation: mean over joints → `[B, H, T, T]`).
        - location-aware MSE against sparse source:
              L = ||attn_tgt[mask] - attn_src[mask]||^2 / |mask|
          where `mask = (attn_src > 0)`.
        - gradient descent on x_t with unit-norm grad and λ = 0.3.
      At every other step: standard DDIM (no guidance, no grad).
4.  Save `[T_valid, J_tgt, 13]` denormalised motion.

Outputs
-------
  eval/results/k_compare/motionclone/pair_<id>_<src>_to_<tgt>.npy
  eval/results/k_compare/motionclone/metrics.json
"""
from __future__ import annotations

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import os
import sys
import json
import time
import argparse
import traceback
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

ROOT = Path(str(PROJECT_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CKPT = 'save/A3_baseline_dataset_truebones_bs_2_latentdim_256/model000229999.pt'
OUT_DIR = ROOT / 'eval/results/k_compare/motionclone'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _stratum_label(p):
    if p['support_same_label'] == 0:
        return 'absent'
    if p['family_gap'] == 'near':
        return 'near_present'
    if p['family_gap'] == 'moderate':
        return 'moderate'
    if p['family_gap'] == 'extreme':
        return 'extreme'
    return 'other'


def _contact_f1(pred, gt, thresh=0.5):
    p = (np.asarray(pred) >= thresh).astype(np.int8).ravel()
    g = (np.asarray(gt) >= thresh).astype(np.int8).ravel()
    n = min(p.size, g.size)
    p, g = p[:n], g[:n]
    tp = int(((p == 1) & (g == 1)).sum())
    fp = int(((p == 1) & (g == 0)).sum())
    fn = int(((p == 0) & (g == 1)).sum())
    pr = tp / (tp + fp + 1e-8); rc = tp / (tp + fn + 1e-8)
    return float(2 * pr * rc / (pr + rc + 1e-8))


def _skating_proxy(motion_13: np.ndarray, contact_groups: dict, tgt_skel: str,
                   cond: dict, fps: int = 30, contact_thresh: float = 0.5) -> float:
    from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np
    try:
        pos = recover_from_bvh_ric_np(motion_13.astype(np.float32))
    except Exception:
        return float('nan')
    T, J, _ = pos.shape
    contact = motion_13[..., 12]
    if contact_groups is None or tgt_skel not in contact_groups:
        foot_joints = np.unique(np.where(contact > contact_thresh)[1]).tolist()
    else:
        foot_joints = []
        for grp in contact_groups[tgt_skel].values():
            foot_joints.extend([int(j) for j in grp if 0 <= int(j) < J])
        foot_joints = sorted(set(foot_joints))
    if not foot_joints:
        return 0.0
    x_vel = np.zeros((T, len(foot_joints)))
    z_vel = np.zeros_like(x_vel)
    if T >= 2:
        dp = pos[1:, foot_joints, :] - pos[:-1, foot_joints, :]
        x_vel[1:] = dp[..., 0]; z_vel[1:] = dp[..., 2]
    horiz = np.sqrt(x_vel ** 2 + z_vel ** 2) * fps
    in_contact = contact[:, foot_joints] > contact_thresh
    if not in_contact.any():
        return 0.0
    return float(horiz[in_contact].mean())


# =========================================================================
# Source-attention extraction
# =========================================================================
def _load_motion_normalised(src_fname: str, src_skel: str, cond: dict,
                            motion_dir: str, max_joints: int,
                            feature_len: int, n_frames: int,
                            device: torch.device):
    """Load source motion [T_raw, J_src, 13] → padded, normalised tensor
    `[1, max_joints, 13, n_frames]`.
    """
    path = os.path.join(motion_dir, src_fname)
    m = np.load(path).astype(np.float32)     # [T_raw, J, 13]
    T_raw, J_src, F = m.shape
    assert F == feature_len
    # crop or pad frames
    T_crop = min(T_raw, n_frames)
    mean = cond[src_skel]['mean'][:J_src]
    std = cond[src_skel]['std'][:J_src] + 1e-6
    m_norm = (m[:T_crop] - mean[None]) / std[None]
    pad = np.zeros((max_joints, feature_len, n_frames), dtype=np.float32)
    pad[:J_src, :, :T_crop] = m_norm.transpose(1, 2, 0)  # [J, 13, T]
    return torch.from_numpy(pad).unsqueeze(0).to(device), J_src, T_crop


def _aggregate_attn(attn_BJH_TT: torch.Tensor, n_joints: int) -> torch.Tensor:
    """Aggregate `[B*J, H, T, T]` over the joint axis (first J rows correspond
    to bs=1 × n_joints joints) → `[B=1, H, T, T]`.
    """
    BJ, H, T, _ = attn_BJH_TT.shape
    # B=1 in all our calls, so BJ == J_padded. Take only the first n_joints
    # (real joints, in the skeleton's native order). This mirrors how
    # `_temporal_mha_block_sin_joint` reshapes x as [T, B*njoints, feats].
    attn = attn_BJH_TT[:n_joints]           # [J, H, T, T]
    return attn.mean(dim=0, keepdim=True)   # [1, H, T, T]


def extract_source_attn(model, diffusion, taps, x_s_tensor, y_src,
                        t_star: int, n_joints_src: int,
                        device: torch.device):
    """Run a single noised-source forward pass and return per-layer
    aggregated attention maps `[1, H, T, T]`.
    """
    from eval.motionclone_attention_taps import (
        clear_last_attn_weights, collect_last_attn_weights,
    )
    clear_last_attn_weights(taps)
    t_tensor = torch.tensor([t_star], device=device, dtype=torch.long)
    noise = torch.randn_like(x_s_tensor)
    x_noisy = diffusion.q_sample(x_s_tensor, t_tensor, noise=noise)
    with torch.no_grad():
        _ = model(x_noisy, t_tensor, y=y_src)
    raw = collect_last_attn_weights(taps)
    agg = [_aggregate_attn(w, n_joints_src) for w in raw]
    clear_last_attn_weights(taps)
    return agg


# =========================================================================
# Target generation with single-step location-aware guidance
# =========================================================================
def generate_with_motionclone(
    model, diffusion, taps, y_tgt, sparse_attn_src: List[torch.Tensor],
    n_joints_tgt: int, mean_t, std_t, max_joints: int, feature_len: int,
    n_frames: int, T_valid: int, T_guide: int,
    n_steps: int, t_star: int, lambda_attn: float,
    device, seed: int,
) -> np.ndarray:
    """DDIM reverse from x_T = N(0, I), inject single-step attention guidance
    at the step whose t_val is closest to `t_star`.
    """
    from eval.motionclone_attention_taps import clear_last_attn_weights

    gen = torch.Generator(device=device).manual_seed(seed)
    x_t = torch.randn((1, max_joints, feature_len, n_frames),
                      device=device, generator=gen)
    alphas_cum = torch.tensor(diffusion.alphas_cumprod, dtype=torch.float32, device=device)
    t_max = diffusion.num_timesteps
    steps = np.linspace(t_max - 1, 0, n_steps + 1).round().astype(int).tolist()
    seen, ordered = set(), []
    for s in steps:
        s = int(s)
        if s not in seen:
            ordered.append(s); seen.add(s)
    steps = ordered
    # Find the step index whose value is closest to t_star.
    guide_i = int(np.argmin(np.abs(np.array(steps) - t_star)))

    for i, t_val in enumerate(steps):
        t_tensor = torch.tensor([t_val], device=device, dtype=torch.long)

        if i == guide_i and lambda_attn > 0:
            # ---- single-step guidance pass ----
            clear_last_attn_weights(taps)
            with torch.enable_grad():
                x_t_req = x_t.detach().requires_grad_(True)
                out = diffusion.p_mean_variance(
                    model, x_t_req, t_tensor, clip_denoised=False,
                    model_kwargs={'y': y_tgt},
                )
                # Target temporal-attn maps (fresh from the tap; leafs grads via x_t_req).
                from eval.motionclone_attention_taps import collect_last_attn_weights
                raw_tgt = collect_last_attn_weights(taps)  # [B*J, H, T, T] each
                attn_tgt = [_aggregate_attn(w, n_joints_tgt) for w in raw_tgt]

                # ---- location-aware semantic loss ----
                loss = torch.zeros((), device=device)
                n_layer = len(sparse_attn_src)
                for attn_s, attn_g in zip(sparse_attn_src, attn_tgt):
                    # align T (both already 121) — truncate to T_guide+1 to skip padding
                    T_lim = min(attn_s.shape[-1], attn_g.shape[-1], T_guide + 1)
                    a_s = attn_s[..., :T_lim, :T_lim]
                    a_g = attn_g[..., :T_lim, :T_lim]
                    mask = (a_s > 0).float()
                    denom = mask.sum().clamp_min(1.0)
                    loss = loss + ((a_g - a_s) ** 2 * mask).sum() / denom
                loss = loss / max(n_layer, 1)

                grad = torch.autograd.grad(loss, x_t_req,
                                           retain_graph=False,
                                           allow_unused=True)[0]
                if grad is not None:
                    g_norm = grad.norm().clamp_min(1e-8)
                    grad = grad / g_norm
                    x_t = (x_t_req - lambda_attn * grad).detach()
                else:
                    x_t = x_t_req.detach()
            clear_last_attn_weights(taps)
            # Re-run without grad to get an up-to-date pred_x0 for DDIM.
            with torch.no_grad():
                out = diffusion.p_mean_variance(
                    model, x_t, t_tensor, clip_denoised=False,
                    model_kwargs={'y': y_tgt},
                )
                x0_hat = out['pred_xstart'].detach()
            # Clear attn again to avoid memory buildup.
            clear_last_attn_weights(taps)
        else:
            # ---- no guidance ----
            with torch.no_grad():
                out = diffusion.p_mean_variance(
                    model, x_t, t_tensor, clip_denoised=False,
                    model_kwargs={'y': y_tgt},
                )
                x0_hat = out['pred_xstart'].detach()
            clear_last_attn_weights(taps)

        # ---- DDIM reverse (eta=0) ----
        if i == len(steps) - 1 or t_val == 0:
            x_t = x0_hat
            break
        t_next = steps[i + 1]
        a_bar_t = alphas_cum[t_val]
        a_bar_next = (alphas_cum[t_next] if t_next >= 0
                      else torch.tensor(1.0, device=device))
        eps = (x_t - torch.sqrt(a_bar_t) * x0_hat) / torch.sqrt(
            (1.0 - a_bar_t).clamp(min=1e-12))
        x_t = (torch.sqrt(a_bar_next) * x0_hat
               + torch.sqrt((1.0 - a_bar_next).clamp(min=0.0)) * eps)
        x_t = x_t.detach()

    # Denorm and crop
    x_ref = x_t[0, :n_joints_tgt].detach().cpu().numpy()   # [J, 13, T]
    x_ref = x_ref.transpose(2, 0, 1)                       # [T, J, 13]
    T_out = min(T_valid, x_ref.shape[0])
    x_ref = x_ref[:T_out]
    mean_np = mean_t[:n_joints_tgt].detach().cpu().numpy()
    std_np = std_t[:n_joints_tgt].detach().cpu().numpy()
    return (x_ref * std_np[None] + mean_np[None]).astype(np.float32)


# =========================================================================
# main runner
# =========================================================================
def run(n_pairs_override=None, n_steps=50, top_k=5, lambda_attn=0.3,
        t_init_frac=0.3, sweep_lambdas=None, sweep_first_n=3):
    # Local imports (after sys.path fix).
    from data_loaders.truebones.truebones_utils.get_opt import get_opt
    from data_loaders.truebones.truebones_utils.param_utils import DATASET_DIR
    from eval.anytop_projection import _load_unconditional_model, _load_t5, _build_y
    from eval.motionclone_attention_taps import install_temporal_taps

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    opt = get_opt(0)
    max_joints = opt.max_joints
    feature_len = opt.feature_len

    cond = np.load(os.path.join(DATASET_DIR, 'cond.npy'),
                   allow_pickle=True).item()
    with open(ROOT / 'eval/quotient_assets/contact_groups.json') as f:
        contact_groups = json.load(f)
    motion_dir = os.path.join(DATASET_DIR, 'motions')

    ckpt_path = str(ROOT / DEFAULT_CKPT)
    model, diffusion, m_args = _load_unconditional_model(ckpt_path, device)
    t5 = _load_t5(m_args.t5_name, device_str=str(device))

    n_frames = m_args.num_frames
    t_max = diffusion.num_timesteps
    t_star = max(1, min(t_max - 1, int(round(t_init_frac * t_max))))
    print(f"[MotionClone] t_star = {t_star} (= {t_init_frac:.2f} × {t_max} steps)")
    print(f"[MotionClone] top_k  = {top_k}   lambda_attn = {lambda_attn}   n_steps = {n_steps}")

    # Install taps once; they remain through all pairs.
    taps = install_temporal_taps(model)
    # Sanity info.
    H = taps[0].num_heads
    print(f"[MotionClone] installed {len(taps)} temporal-attention taps "
          f"(H={H}); params / gradients unchanged.")

    with open(ROOT / 'idea-stage/eval_pairs.json') as f:
        pairs_all = json.load(f)['pairs']
    pairs = pairs_all if n_pairs_override is None else pairs_all[:n_pairs_override]

    per_pair = []
    wall_t0 = time.time()
    gpu_peak_mem_gb = 0.0

    y_cache = {}  # skel -> (y, n_joints, mean, std)
    def _get_y(skel):
        if skel not in y_cache:
            y, nj, mean_np, std_np = _build_y(
                skel, cond, t5, n_frames,
                m_args.temporal_window, max_joints, device)
            y_cache[skel] = (y, nj, mean_np, std_np)
        return y_cache[skel]

    # Optional sweep: run a few λ values on the first `sweep_first_n` pairs.
    sweep_results = {}
    if sweep_lambdas:
        print(f"\n[MotionClone] running λ-sweep {sweep_lambdas} on pairs 0..{sweep_first_n-1}")
        for lam in sweep_lambdas:
            sweep_results[str(lam)] = []
            for p in pairs_all[:sweep_first_n]:
                pid = p['pair_id']
                try:
                    rec = _run_single(p, cond, contact_groups, motion_dir, t5,
                                      model, diffusion, taps, max_joints,
                                      feature_len, n_frames, t_star, top_k,
                                      lam, n_steps, device, _get_y,
                                      save_output=False)
                    sweep_results[str(lam)].append(rec)
                    print(f"    λ={lam}  pair_{pid:02d}: skate={rec.get('skating_proxy', 'nan'):.4f}  "
                          f"contact_f1_src={rec.get('contact_f1_vs_source', float('nan')):.3f}  "
                          f"gen_t={rec.get('gen_time_sec', 0):.1f}s")
                except Exception as e:
                    sweep_results[str(lam)].append({'pair_id': pid, 'error': str(e)})

    # Main 30-pair run at `lambda_attn`.
    print(f"\n[MotionClone] 30-pair run at λ={lambda_attn}")
    for p in pairs:
        pid = p['pair_id']
        src_skel = p['source_skel']; tgt_skel = p['target_skel']
        src_fname = p['source_fname']
        print(f"\n=== pair {pid:02d}  {src_skel}({src_fname}) -> {tgt_skel}  "
              f"gap={p['family_gap']} supp={p['support_same_label']} ===", flush=True)

        try:
            rec = _run_single(p, cond, contact_groups, motion_dir, t5,
                              model, diffusion, taps, max_joints,
                              feature_len, n_frames, t_star, top_k,
                              lambda_attn, n_steps, device, _get_y,
                              save_output=True)
            try:
                gpu_peak_mem_gb = max(gpu_peak_mem_gb,
                                      torch.cuda.max_memory_allocated(device) / 1e9)
            except Exception:
                pass
        except Exception as e:
            rec = {
                'pair_id': pid, 'source_skel': src_skel, 'target_skel': tgt_skel,
                'source_fname': src_fname,
                'family_gap': p['family_gap'],
                'support_same_label': int(p['support_same_label']),
                'stratum': _stratum_label(p),
                'status': 'failed',
                'error': str(e) + '\n' + traceback.format_exc(limit=3),
            }
            print(f"  FAILED: {e}", flush=True)
        per_pair.append(rec)

    total_time = time.time() - wall_t0

    # Stratified means.
    strata_order = ['near_present', 'absent', 'moderate', 'extreme']
    strata = {s: [] for s in strata_order}
    for r in per_pair:
        strata[r.get('stratum', 'other')].append(r)

    def _sstats(entries):
        ok = [e for e in entries if e.get('status') == 'ok']
        if not ok:
            return {'n_total': len(entries), 'n_ok': 0}
        def mean_of(k):
            vals = [e.get(k, float('nan')) for e in ok]
            vals = [v for v in vals if isinstance(v, (int, float)) and np.isfinite(v)]
            return float(np.mean(vals)) if vals else float('nan')
        return {
            'n_total': len(entries),
            'n_ok': len(ok),
            'contact_f1_vs_source': mean_of('contact_f1_vs_source'),
            'skating_proxy': mean_of('skating_proxy'),
            'gen_time_sec': mean_of('gen_time_sec'),
            'pair_runtime_sec': mean_of('pair_runtime_sec'),
        }

    strata_stats = {s: _sstats(strata[s]) for s in strata_order}
    out = {
        'method': 'motionclone',
        'mechanism': 'sparse_single_step_location_aware',
        'ckpt': ckpt_path,
        'top_k': int(top_k),
        'lambda_attn': float(lambda_attn),
        'n_steps': int(n_steps),
        't_init_frac': float(t_init_frac),
        't_star': int(t_star),
        't_max': int(t_max),
        'n_decoder_layers': len(taps),
        'heads_per_layer': int(H),
        'total_runtime_sec': float(total_time),
        'gpu_peak_mem_gb': float(gpu_peak_mem_gb),
        'n_pairs': len(pairs),
        'n_ok': sum(1 for r in per_pair if r.get('status') == 'ok'),
        'n_failed': sum(1 for r in per_pair if r.get('status') == 'failed'),
        'strata_stats': strata_stats,
        'per_pair': per_pair,
        'sweep_results': sweep_results if sweep_lambdas else None,
    }
    with open(OUT_DIR / 'metrics.json', 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n=== DONE: total {total_time:.1f}s, n_ok={out['n_ok']}/{out['n_pairs']}, "
          f"failed={out['n_failed']}, peak GPU = {gpu_peak_mem_gb:.2f} GB ===")
    print(f"metrics saved to {OUT_DIR / 'metrics.json'}")
    return out


def _run_single(p, cond, contact_groups, motion_dir, t5, model, diffusion,
                taps, max_joints, feature_len, n_frames, t_star, top_k,
                lambda_attn, n_steps, device, _get_y, save_output=True):
    """Run MotionClone for one pair. Returns the per-pair record dict."""
    from data_loaders.truebones.truebones_utils.get_opt import get_opt  # noqa: F401
    from eval.motionclone_attention_taps import sparsify_topk, interpolate_time
    from eval.quotient_extractor import extract_quotient

    pid = p['pair_id']
    src_skel = p['source_skel']; tgt_skel = p['target_skel']
    src_fname = p['source_fname']
    rec = {
        'pair_id': pid, 'source_skel': src_skel, 'target_skel': tgt_skel,
        'source_fname': src_fname,
        'family_gap': p['family_gap'],
        'support_same_label': int(p['support_same_label']),
        'stratum': _stratum_label(p),
        'status': 'pending',
    }
    t0 = time.time()

    # Source.
    x_s, J_src, T_src = _load_motion_normalised(
        src_fname, src_skel, cond, motion_dir,
        max_joints, feature_len, n_frames, device)
    y_src, _, _, _ = _get_y(src_skel)

    # Target.
    y_tgt, J_tgt, mean_np, std_np = _get_y(tgt_skel)
    mean_pad = np.zeros((max_joints, feature_len), dtype=np.float32)
    std_pad = np.ones((max_joints, feature_len), dtype=np.float32)
    mean_pad[:J_tgt] = mean_np; std_pad[:J_tgt] = std_np
    mean_t = torch.from_numpy(mean_pad).to(device)
    std_t = torch.from_numpy(std_pad).to(device)

    # --- source attention extraction (single step) ---
    t1 = time.time()
    src_attn_raw = extract_source_attn(
        model, diffusion, taps, x_s, y_src,
        t_star=t_star, n_joints_src=J_src, device=device,
    )
    # Sparsify each layer's map; restrict to the valid temporal extent on the
    # source side (T_src + 1 to include the tpos frame index 0).
    src_attn_sparse = []
    for a in src_attn_raw:
        T_now = a.shape[-1]
        # model's temporal dim = n_frames + 1 → valid = T_src + 1 (cap at T_now)
        v_src = min(T_now, T_src + 1)
        sp = sparsify_topk(a, k=top_k, valid_len=v_src)
        src_attn_sparse.append(sp)
    src_extract_t = time.time() - t1

    # --- target generation with guidance ---
    # Valid target temporal extent: min(T_src, n_frames) — we keep the same
    # number of frames the source had; motion_13 output gets cropped to this.
    T_valid = int(min(T_src, n_frames))
    # Guidance window: only supervise within the valid overlap.
    T_guide = T_valid
    gen_t0 = time.time()
    motion_denorm = generate_with_motionclone(
        model, diffusion, taps, y_tgt, src_attn_sparse,
        n_joints_tgt=J_tgt, mean_t=mean_t, std_t=std_t,
        max_joints=max_joints, feature_len=feature_len,
        n_frames=n_frames, T_valid=T_valid, T_guide=T_guide,
        n_steps=n_steps, t_star=t_star, lambda_attn=lambda_attn,
        device=device, seed=pid,
    )
    gen_t = time.time() - gen_t0

    # --- metrics ---
    # Source contact schedule (cached in a simple local dict).
    try:
        q_src = extract_quotient(src_fname, cond[src_skel],
                                 contact_groups=contact_groups,
                                 motion_dir=motion_dir)
        src_sched_agg = np.asarray(q_src['contact_sched']).sum(axis=1)
    except Exception:
        src_sched_agg = None

    if src_sched_agg is not None:
        pred_contact = (motion_denorm[..., 12] > 0.5).astype(np.float32).sum(axis=1)
        Ts = src_sched_agg.shape[0]; Tp = pred_contact.shape[0]
        if Ts >= 4 and Tp >= 4:
            idx = np.clip(np.linspace(0, Ts - 1, Tp).astype(int), 0, Ts - 1)
            rec['contact_f1_vs_source'] = _contact_f1(pred_contact, src_sched_agg[idx])
        else:
            rec['contact_f1_vs_source'] = float('nan')
    else:
        rec['contact_f1_vs_source'] = float('nan')

    rec['skating_proxy'] = _skating_proxy(motion_denorm, contact_groups, tgt_skel, cond)
    rec['n_joints_src'] = int(J_src); rec['n_joints_tgt'] = int(J_tgt)
    rec['T_src'] = int(T_src); rec['T_out'] = int(motion_denorm.shape[0])
    rec['src_extract_time_sec'] = float(src_extract_t)
    rec['gen_time_sec'] = float(gen_t)
    rec['pair_runtime_sec'] = float(time.time() - t0)
    rec['status'] = 'ok'

    if save_output:
        out_path = OUT_DIR / f"pair_{pid:02d}_{src_skel}_to_{tgt_skel}.npy"
        np.save(out_path, motion_denorm)
        rec['output_path'] = str(out_path)
        print(f"  ok: contact_f1_vs_src={rec['contact_f1_vs_source']:.3f}  "
              f"skate={rec['skating_proxy']:.4f}  "
              f"gen_t={gen_t:.1f}s  total_t={rec['pair_runtime_sec']:.1f}s", flush=True)

    return rec


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_pairs', type=int, default=None)
    ap.add_argument('--n_steps', type=int, default=50)
    ap.add_argument('--top_k', type=int, default=5)
    ap.add_argument('--lambda_attn', type=float, default=0.3)
    ap.add_argument('--t_init_frac', type=float, default=0.3)
    ap.add_argument('--sweep_lambdas', type=str, default=None,
                    help='comma-separated list, e.g. "0.1,0.3,1.0"')
    ap.add_argument('--sweep_first_n', type=int, default=3)
    args = ap.parse_args()
    sw = None
    if args.sweep_lambdas:
        sw = [float(x) for x in args.sweep_lambdas.split(',') if x.strip()]
    run(n_pairs_override=args.n_pairs, n_steps=args.n_steps,
        top_k=args.top_k, lambda_attn=args.lambda_attn,
        t_init_frac=args.t_init_frac,
        sweep_lambdas=sw, sweep_first_n=args.sweep_first_n)
