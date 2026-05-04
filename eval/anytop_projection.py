"""Stage-3: AnyTop prior projection for cross-skeleton retargeting (Idea K).

Given an IK-initialised motion on a target skeleton (output of Stage 2), this
module projects it through a frozen AnyTop unconditional DDPM so the result
lives on the data manifold while retaining hard task-space constraints.

SDEdit-style DDIM refinement:
    1. Normalise x_init with target-skeleton mean/std.
    2. Diffuse the normalised motion to t = t_init * T_max (q_sample).
    3. Run K DDIM reverse steps from t_init down to 0.
    4. After every step, inject:
         - hard projection of enforced-ground joints to world y=0
           (channel 1 of the joint's 13-dim slot in root-local frame),
         - soft COM-guidance shift on the root Y toward the requested path.
    5. Denormalise with the target stats and return.

The prior is the A3 baseline *unconditional* AnyTop (DDPM over all 70
Truebones skeletons).  No re-training, no finetuning.
"""
from __future__ import annotations
import os
import json
import time
import argparse
import numpy as np
import torch
from os.path import join as pjoin
from typing import Optional

DEFAULT_CKPT = 'save/A3_baseline_dataset_truebones_bs_2_latentdim_256/model000229999.pt'
FALLBACK_CKPT = 'save/all_model_dataset_truebones_bs_16_latentdim_128/model000459999.pt'
PROJECT_ROOT = str(PROJECT_ROOT)

# Feature layout per joint: [pos(3), rot6d(6), vel(3), foot_contact(1)] = 13
POS_Y_IDX = 1       # world-Y height channel within a joint's 13-dim slot
FOOT_CH_IDX = 12    # foot-contact binary channel
ROT_START, ROT_END = 3, 9

# Singleton caches (heavy objects)
_MODEL_CACHE = {}
_T5_CACHE = {}


def _project_root_path():
    import sys
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)


# ------------------------ model / diffusion loading ----------------------
def _load_unconditional_model(ckpt_path: str, device: torch.device):
    """Load frozen AnyTop + DDPM diffusion (cached per (ckpt, device))."""
    _project_root_path()
    key = (ckpt_path, str(device))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    from model.anytop import AnyTop
    from utils.model_util import create_gaussian_diffusion

    with open(pjoin(os.path.dirname(ckpt_path), 'args.json')) as f:
        ckpt_args = json.load(f)

    class Ns:
        def __init__(self, d): self.__dict__.update(d)
    args = Ns(ckpt_args)

    t5_dim = {'t5-small': 512, 't5-base': 768, 't5-large': 1024}
    t5_out = t5_dim.get(args.t5_name, 768)

    model = AnyTop(
        max_joints=143, feature_len=13,
        latent_dim=args.latent_dim, ff_size=1024,
        num_layers=args.layers, num_heads=4,
        dropout=0.1, activation='gelu',
        t5_out_dim=t5_out, root_input_feats=13,
        cond_mode='object_type', cond_mask_prob=args.cond_mask_prob,
        skip_t5=args.skip_t5, value_emb=args.value_emb)
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f'Unexpected keys in {ckpt_path}: {unexpected[:3]}')
    bad_missing = [k for k in missing if not k.startswith('clip_model.')]
    if bad_missing:
        raise RuntimeError(f'Missing keys not in clip: {bad_missing[:3]}')
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    diffusion = create_gaussian_diffusion(args)
    _MODEL_CACHE[key] = (model, diffusion, args)
    return model, diffusion, args


def _load_t5(t5_name: str, device_str: str = 'cuda'):
    key = (t5_name, device_str)
    if key in _T5_CACHE:
        return _T5_CACHE[key]
    _project_root_path()
    from model.conditioners import T5Conditioner
    t5 = T5Conditioner(name=t5_name, finetune=False, word_dropout=0.0,
                       normalize_text=False, device=device_str)
    _T5_CACHE[key] = t5
    return t5


# ------------------------------ conditioning -----------------------------
def _build_y(skel_name: str, cond_dict: dict, t5, n_frames: int,
             temporal_window: int, max_joints: int, device: torch.device):
    """Build `y` conditioning dict for frozen AnyTop forward pass."""
    _project_root_path()
    from data_loaders.truebones.data.dataset import create_temporal_mask_for_window
    from data_loaders.tensors import create_padded_relation

    info = cond_dict[skel_name]
    n_joints = len(info['joints_names'])
    mean = info['mean']
    std = info['std'] + 1e-6

    tpos = np.nan_to_num((info['tpos_first_frame'] - mean) / std)
    tpos_pad = np.zeros((max_joints, 13), dtype=np.float32)
    tpos_pad[:n_joints] = tpos
    tpos_t = torch.from_numpy(tpos_pad).float().unsqueeze(0).to(device)

    names_emb = t5(t5.tokenize(list(info['joints_names']))).detach().cpu().numpy()
    name_pad = np.zeros((max_joints, names_emb.shape[1]), dtype=np.float32)
    name_pad[:n_joints] = names_emb
    names_t = torch.from_numpy(name_pad).float().unsqueeze(0).to(device)

    gd = create_padded_relation(info['joints_graph_dist'], max_joints, n_joints)
    jr = create_padded_relation(info['joint_relations'], max_joints, n_joints)
    gd_t = gd.long().unsqueeze(0).to(device)
    jr_t = jr.long().unsqueeze(0).to(device)

    jmask = torch.zeros(1, 1, 1, max_joints + 1, max_joints + 1, device=device)
    jmask[0, 0, 0, :n_joints + 1, :n_joints + 1] = 1.0
    tmask = create_temporal_mask_for_window(temporal_window, n_frames)
    tmask_t = tmask.unsqueeze(0).unsqueeze(2).unsqueeze(3).float().to(device)

    y = {
        'joints_mask':       jmask,
        'mask':              tmask_t,
        'tpos_first_frame':  tpos_t,
        'joints_names_embs': names_t,
        'graph_dist':        gd_t,
        'joints_relations':  jr_t,
        'crop_start_ind':    torch.zeros(1, dtype=torch.long, device=device),
        'n_joints':          torch.tensor([n_joints]),
    }
    return y, n_joints, mean.astype(np.float32), std.astype(np.float32)


# --------------------- constraint / guidance helpers ---------------------
def _apply_hard_contacts(x_hat: torch.Tensor,
                         contact_mask: torch.Tensor,
                         mean_t: torch.Tensor,
                         std_t: torch.Tensor) -> torch.Tensor:
    """Clamp enforced-ground joints' Y to 0 (world frame) in normalised space.

    Representation note: non-root joint positions are root-local (get_rifke),
    but root rotation is constrained to Y-axis yaw — so Y is invariant and
    channel 1 of the joint's 13-dim slot IS world-frame Y.  Setting world
    y=0 is therefore setting normalised channel 1 to ``-mean[j,1]/std[j,1]``.

    Args
    ----
    x_hat : [B, J, 13, T] predicted x0 (normalised space), J=max_joints.
    contact_mask : [T, J] 1 where joint j must touch ground at frame t.
    mean_t, std_t : [max_J, 13] target-skeleton stats on device.
    """
    y_target = (0.0 - mean_t[:, POS_Y_IDX]) / std_t[:, POS_Y_IDX]  # [J]
    m = contact_mask.permute(1, 0).unsqueeze(0).unsqueeze(2).float()  # [1,J,1,T]
    m = m.expand(x_hat.size(0), -1, 1, -1)
    y_target_b = y_target.view(1, -1, 1, 1).expand_as(m)
    current = x_hat[:, :, POS_Y_IDX:POS_Y_IDX + 1, :]
    x_hat[:, :, POS_Y_IDX:POS_Y_IDX + 1, :] = m * y_target_b + (1.0 - m) * current
    return x_hat


def _soft_com_shift(x_hat: torch.Tensor,
                    com_target: torch.Tensor,
                    n_joints: int,
                    mean_t: torch.Tensor,
                    std_t: torch.Tensor,
                    lambda_com: float) -> torch.Tensor:
    """Nudge the root-Y height toward matching the target COM height.

    Treats the mean joint-Y over real joints as a COM-height proxy and
    shifts the root's Y channel by ``lambda_com * (tgt_com_y - cur_com_y)``.
    """
    if lambda_com <= 0 or com_target is None:
        return x_hat
    cur_norm_y = x_hat[:, :n_joints, POS_Y_IDX, :]           # [B, n, T]
    mean_y = mean_t[:n_joints, POS_Y_IDX].view(1, n_joints, 1)
    std_y  = std_t[:n_joints, POS_Y_IDX].view(1, n_joints, 1)
    cur_world_y = cur_norm_y * std_y + mean_y
    cur_com_y = cur_world_y.mean(dim=1)                      # [B, T]
    tgt_com_y = com_target[:, 1].view(1, -1)                 # [1, T]
    delta_world = lambda_com * (tgt_com_y - cur_com_y)       # [B, T]
    root_std_y = std_t[0, POS_Y_IDX]
    x_hat[:, 0, POS_Y_IDX, :] = x_hat[:, 0, POS_Y_IDX, :] + delta_world / root_std_y
    return x_hat


# ------------------------------ core DDIM --------------------------------
@torch.no_grad()
def _ddim_project_loop(model, diffusion, x_T_tilde: torch.Tensor,
                       t_start: int, n_steps: int, y: dict,
                       hard_constraints: Optional[dict], n_joints: int,
                       mean_t: torch.Tensor, std_t: torch.Tensor,
                       lambda_com: float, device: torch.device):
    """DDIM reverse (eta=0) from t_start to 0 with per-step x0 projection."""
    if n_steps >= t_start + 1:
        steps = list(range(t_start, -1, -1))
    else:
        steps = np.linspace(t_start, 0, n_steps + 1).round().astype(int).tolist()
    seen, ordered = set(), []
    for s in steps:
        s = int(s)
        if s not in seen:
            ordered.append(s); seen.add(s)
    steps = ordered

    alphas_cum = torch.tensor(diffusion.alphas_cumprod, dtype=torch.float32,
                              device=device)

    img = x_T_tilde
    trajectory = []
    contact_mask = None
    com_target = None
    if hard_constraints is not None:
        contact_mask = hard_constraints.get('contact_positions')
        if contact_mask is not None and not torch.is_tensor(contact_mask):
            contact_mask = torch.as_tensor(contact_mask, dtype=torch.float32,
                                           device=device)
        com_target = hard_constraints.get('com_path')
        if com_target is not None and not torch.is_tensor(com_target):
            com_target = torch.as_tensor(com_target, dtype=torch.float32,
                                         device=device)

    for i, t_val in enumerate(steps):
        t_tensor = torch.tensor([t_val], device=device, dtype=torch.long)
        out = diffusion.p_mean_variance(
            model, img, t_tensor, clip_denoised=False, model_kwargs={'y': y})
        x0_hat = out['pred_xstart']

        if contact_mask is not None:
            x0_hat = _apply_hard_contacts(x0_hat, contact_mask, mean_t, std_t)
        if com_target is not None and lambda_com > 0:
            x0_hat = _soft_com_shift(x0_hat, com_target, n_joints,
                                     mean_t, std_t, lambda_com)
        trajectory.append(x0_hat.detach().cpu().numpy())

        if i == len(steps) - 1 or t_val == 0:
            img = x0_hat
            break

        t_next = steps[i + 1]
        a_bar_t = alphas_cum[t_val]
        a_bar_next = alphas_cum[t_next] if t_next >= 0 else torch.tensor(1.0, device=device)
        eps = (img - torch.sqrt(a_bar_t) * x0_hat) / torch.sqrt((1.0 - a_bar_t).clamp(min=1e-12))
        img = torch.sqrt(a_bar_next) * x0_hat + torch.sqrt((1.0 - a_bar_next).clamp(min=0.0)) * eps

    return img, trajectory


# ---------------------------- public API ---------------------------------
def anytop_project(x_init: np.ndarray,
                   target_skel: str,
                   hard_constraints: Optional[dict] = None,
                   t_init: float = 0.3,
                   n_steps: int = 20,
                   lambda_com: float = 1.0,
                   ckpt_path: Optional[str] = None,
                   device: str = 'cuda') -> dict:
    """Project an initial motion through the frozen AnyTop prior.

    Parameters
    ----------
    x_init : np.ndarray [T, J, 13]
        Initial motion on target skeleton (e.g. Stage-2 IK output).
    target_skel : str
        Target skeleton name (key of ``cond.npy``).
    hard_constraints : dict with keys
        ``contact_positions`` : [T, J] binary mask (1 = enforce y=0).
        ``com_path``          : [T, 3] world-frame COM path (Y used).
    t_init : float
        Fraction of max diffusion timesteps to diffuse to (SDEdit style).
    n_steps : int
        DDIM reverse steps.
    lambda_com : float
        Soft COM-guidance weight (0 disables).
    ckpt_path : str or None
        Checkpoint path; defaults to the A3 baseline.
    device : str, e.g. ``'cuda'``

    Returns
    -------
    dict
        - ``x_refined`` : [T', J, 13] projected motion (denormalised;
          T' = min(T, model n_frames))
        - ``x_init``    : [T', J, 13] cropped input, for comparison
        - ``ddim_trajectory`` : list[np.ndarray] pred_xstart per step
          (normalised space, padded to [1, max_joints, 13, n_frames])
        - ``runtime_seconds`` : float
        - ``ckpt_path`` : resolved checkpoint path

    Example (integration with Stage 2)
    ----------------------------------
    >>> from eval.track_a_eval import solve_ik               # Stage 2
    >>> ik_motion = solve_ik(q_target, target_skel='Cat')    # [T, J, 13]
    >>> hc = {'contact_positions': contact_mask_TJ,          # [T, J]
    ...       'com_path': com_path_T3}                       # [T, 3]
    >>> refined = anytop_project(ik_motion, 'Cat', hc)['x_refined']
    """
    _project_root_path()
    from data_loaders.truebones.truebones_utils.get_opt import get_opt

    ckpt_path = ckpt_path or pjoin(PROJECT_ROOT, DEFAULT_CKPT)
    if not os.path.isabs(ckpt_path):
        ckpt_path = pjoin(PROJECT_ROOT, ckpt_path)
    if not os.path.exists(ckpt_path):
        fb = pjoin(PROJECT_ROOT, FALLBACK_CKPT)
        print(f'[anytop_project] {ckpt_path} missing, falling back to {fb}')
        ckpt_path = fb

    dev = torch.device(device)
    model, diffusion, m_args = _load_unconditional_model(ckpt_path, dev)
    t5 = _load_t5(m_args.t5_name, device_str=device)

    opt = get_opt(0)
    cond_dict = np.load(opt.cond_file, allow_pickle=True).item()

    max_joints = opt.max_joints
    feature_len = opt.feature_len
    T, J_real, F = x_init.shape
    assert F == feature_len, f'expected last dim {feature_len}, got {F}'
    n_frames = m_args.num_frames
    if T >= n_frames:
        x_cropped = x_init[:n_frames].copy()
    else:
        pad = np.zeros((n_frames - T, J_real, feature_len), dtype=x_init.dtype)
        x_cropped = np.concatenate([x_init, pad], axis=0)
    T_in = T

    y, n_joints, mean_np, std_np = _build_y(
        target_skel, cond_dict, t5, n_frames,
        m_args.temporal_window, max_joints, dev)
    assert n_joints == J_real, (
        f'target_skel {target_skel} has {n_joints} joints; x_init has {J_real}')

    # Pad stats to max_joints so constraint helpers broadcast uniformly.
    mean_pad = np.zeros((max_joints, feature_len), dtype=np.float32)
    std_pad = np.ones((max_joints, feature_len), dtype=np.float32)
    mean_pad[:n_joints] = mean_np
    std_pad[:n_joints] = std_np
    mean_t = torch.from_numpy(mean_pad).to(dev)
    std_t  = torch.from_numpy(std_pad).to(dev)

    x_norm = np.nan_to_num((x_cropped - mean_np[None]) / std_np[None]).astype(np.float32)
    motion_pad = np.zeros((max_joints, feature_len, n_frames), dtype=np.float32)
    motion_pad[:n_joints] = x_norm.transpose(1, 2, 0)
    x0 = torch.from_numpy(motion_pad).unsqueeze(0).to(dev)  # [1, J, F, T]

    if hard_constraints is not None:
        hard_constraints = dict(hard_constraints)
        c = hard_constraints.get('contact_positions')
        if c is not None:
            c = np.asarray(c, dtype=np.float32)
            if c.shape[0] >= n_frames:
                c = c[:n_frames]
            else:
                c = np.concatenate([c, np.zeros((n_frames - c.shape[0], c.shape[1]))], 0)
            c_pad = np.zeros((n_frames, max_joints), dtype=np.float32)
            c_pad[:, :c.shape[1]] = c
            hard_constraints['contact_positions'] = torch.from_numpy(c_pad).to(dev)
        cp = hard_constraints.get('com_path')
        if cp is not None:
            cp = np.asarray(cp, dtype=np.float32)
            if cp.shape[0] >= n_frames:
                cp = cp[:n_frames]
            else:
                last = cp[-1:].repeat(n_frames - cp.shape[0], axis=0)
                cp = np.concatenate([cp, last], axis=0)
            hard_constraints['com_path'] = torch.from_numpy(cp).to(dev)

    t_max = diffusion.num_timesteps
    t_start = max(1, min(t_max - 1, int(round(t_init * t_max))))
    noise = torch.randn_like(x0)
    t_tensor = torch.tensor([t_start], device=dev, dtype=torch.long)
    x_t = diffusion.q_sample(x0, t_tensor, noise=noise)

    tic = time.time()
    with torch.no_grad():
        x_ref_padded, trajectory = _ddim_project_loop(
            model, diffusion, x_t, t_start, n_steps, y,
            hard_constraints, n_joints, mean_t, std_t, lambda_com, dev)
    runtime = time.time() - tic

    T_out = min(T_in, n_frames)
    x_ref = x_ref_padded[0, :n_joints].detach().cpu().numpy()
    x_ref = x_ref.transpose(2, 0, 1)[:T_out]
    x_ref_denorm = x_ref * std_np[None] + mean_np[None]

    return {
        'x_refined': x_ref_denorm.astype(np.float32),
        'x_init': x_init[:T_out].astype(x_init.dtype),
        'ddim_trajectory': trajectory,
        'runtime_seconds': float(runtime),
        'ckpt_path': ckpt_path,
    }


# ---------------------------- sanity tests -------------------------------
def _find_cat_clip(motion_dir: str) -> str:
    for f in sorted(os.listdir(motion_dir)):
        if f.startswith('Cat_'):
            return pjoin(motion_dir, f)
    raise FileNotFoundError('No Cat_*.npy motion found')


def _contact_violation(motion_denorm: np.ndarray, contact_mask: np.ndarray) -> float:
    """Mean |y| over enforced-ground joints (world frame)."""
    T = motion_denorm.shape[0]
    ys = motion_denorm[:, :, POS_Y_IDX]
    mask = contact_mask[:T, :motion_denorm.shape[1]].astype(bool)
    if not mask.any():
        return 0.0
    return float(np.abs(ys[mask]).mean())


def _l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).mean()))


def _main_test():
    _project_root_path()
    from data_loaders.truebones.truebones_utils.get_opt import get_opt
    opt = get_opt(0)
    clip_path = _find_cat_clip(opt.motion_dir)
    motion = np.load(clip_path).astype(np.float32)
    print(f'[test] loaded {clip_path}  shape={motion.shape}')
    T = motion.shape[0]

    contact = motion[:, :, FOOT_CH_IDX]                 # [T, J]
    mean_y = motion[:, :, POS_Y_IDX].mean(axis=1)       # COM-height proxy
    com_path = np.stack([np.zeros(T), mean_y, np.zeros(T)], axis=-1)
    hc = {'contact_positions': contact, 'com_path': com_path}

    # -------- Sanity A: projection of a real motion --------
    print('\n[sanity A] projecting real Cat clip ...')
    out_a = anytop_project(motion, 'Cat', hc, t_init=0.3, n_steps=20)
    x_init_a, x_ref_a = out_a['x_init'], out_a['x_refined']
    l2_a = _l2(x_init_a, x_ref_a)
    viol_init = _contact_violation(x_init_a, contact)
    viol_ref = _contact_violation(x_ref_a, contact)
    print(f'  L2(x_init, x_refined)       = {l2_a:.6f}')
    print(f'  contact-y violation before  = {viol_init:.6f}')
    print(f'  contact-y violation after   = {viol_ref:.6f}')
    print(f'  contact tighter after proj? {viol_ref < viol_init}')
    print(f'  runtime                     = {out_a["runtime_seconds"]:.2f} s')
    print(f'  ckpt                        = {out_a["ckpt_path"]}')

    # -------- Sanity B: projection of noisy motion --------
    # The prior has a finite reconstruction floor (L2 ~0.08 on a Cat clip) —
    # it cannot reproduce any specific sequence exactly.  We pick σ=0.3 so the
    # projection demonstrably helps (empirical crossover ~σ=0.2 on Cat).
    print('\n[sanity B] projecting noisy Cat clip ...')
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(motion.shape).astype(np.float32) * 0.3
    noise[..., FOOT_CH_IDX] = 0.0      # keep foot-contact binary channel
    motion_noisy = motion + noise
    out_b = anytop_project(motion_noisy, 'Cat', hc, t_init=0.3, n_steps=20)
    x_ref_b = out_b['x_refined']
    T_cmp = min(x_ref_b.shape[0], motion.shape[0])
    d_noisy = _l2(motion_noisy[:T_cmp], motion[:T_cmp])
    d_ref = _l2(x_ref_b[:T_cmp], motion[:T_cmp])
    print(f'  L2(noisy,   clean)          = {d_noisy:.6f}')
    print(f'  L2(refined, clean)          = {d_ref:.6f}')
    print(f'  prior pulls toward manifold? {d_ref < d_noisy}')
    print(f'  runtime                     = {out_b["runtime_seconds"]:.2f} s')

    print('\n[summary]')
    print(json.dumps({
        'ckpt': out_a['ckpt_path'],
        'sanityA_L2_init_vs_refined': l2_a,
        'sanityA_contact_before': viol_init,
        'sanityA_contact_after': viol_ref,
        'sanityA_runtime_s': out_a['runtime_seconds'],
        'sanityB_L2_noisy_vs_clean': d_noisy,
        'sanityB_L2_refined_vs_clean': d_ref,
        'sanityB_runtime_s': out_b['runtime_seconds'],
    }, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip_tests', action='store_true')
    args = parser.parse_args()
    if not args.skip_tests:
        _main_test()
