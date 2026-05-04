"""Constrained-IK solver fork supporting optional ``theta_init`` warm-start.

Mirrors ``eval.ik_solver.solve_ik`` exactly, but adds one extra kwarg:

    theta_init : Optional[np.ndarray]   # [T, J, 3] axis-angle seed

When supplied, the optimiser's ``theta`` parameter is initialised from
``theta_init`` instead of zeros.  The root-yaw seeding (from heading of
``q_star['com_path']``) is still performed on top of the initial theta for
joint 0; this mirrors the original behaviour so the only difference is the
non-root joints' seed rotations.

Every other detail (losses, weights, Adam LR schedule, grad clipping,
theta box-projection, final best-loss tracking, reconstruction helpers) is
imported from ``eval.ik_solver`` unchanged.  This keeps the fork minimal
and guarantees metric parity.
"""
from __future__ import annotations
import math
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from eval.ik_solver import (  # noqa: F401 (re-export for convenience)
    _rodrigues,
    fk_torch,
    _subtree_masses_np,
    _body_scale_np,
    _weighted_subtree,
    _compute_rest_root_height,
    _forward_from_centroids_torch,
    _heading_velocity_torch,
    _cadence_soft_torch,
    _soft_contacts_torch,
    _init_root_pos_from_q,
    _reconstruct_quotient,
    _q_component_errors,
    DEFAULT_WEIGHTS,
)

_EPS = 1e-8


def solve_ik_warmstart(q_star: dict,
                       target_skel_cond: dict,
                       contact_groups: dict,
                       n_iters: int = 400,
                       weights: Optional[dict] = None,
                       verbose: bool = False,
                       fps: int = 30,
                       device: Optional[str] = None,
                       theta_init: Optional[np.ndarray] = None,
                       track_convergence: bool = False) -> dict:
    """Same as ``eval.ik_solver.solve_ik`` but accepts ``theta_init``.

    Extra kwargs
    ------------
    theta_init : [T, J, 3] float32 or None
        If supplied, seed ``theta`` from this axis-angle array instead of zeros.
        Shape mismatches are handled by truncation / zero-padding on the time
        and joint axes so a source-skeleton warm-start cannot silently crash.
    track_convergence : bool
        If True, the returned dict also contains ``loss_trace`` : list[float]
        of the total loss value per iteration (useful for convergence plots).
    """
    device = torch.device(device) if device is not None else torch.device('cpu')
    w_cfg = {**DEFAULT_WEIGHTS, **(weights or {})}

    offsets_np = np.asarray(target_skel_cond['offsets'], dtype=np.float32)
    J = offsets_np.shape[0]
    parents = [int(p) for p in target_skel_cond['parents']][:J]
    for j in range(J):
        if parents[j] >= j:
            parents[j] = -1 if j == 0 else 0
    com_path = np.asarray(q_star['com_path'], dtype=np.float32)
    T = com_path.shape[0]
    body_scale = _body_scale_np(offsets_np)
    rest_root_h = _compute_rest_root_height(parents, offsets_np)

    com_path_t = torch.tensor(com_path, dtype=torch.float32, device=device)
    heading_tgt = torch.tensor(np.asarray(q_star['heading_vel'], dtype=np.float32),
                               device=device)
    cadence_tgt = float(q_star.get('cadence', 0.0))
    sched_tgt = torch.tensor(np.asarray(q_star['contact_sched'], dtype=np.float32),
                             device=device)
    if sched_tgt.dim() == 1:
        sched_tgt = sched_tgt.unsqueeze(-1)

    group_names = q_star.get('contact_group_names') or sorted(contact_groups.keys())
    group_idx_tgt: List[List[int]] = []
    for name in group_names:
        if name in contact_groups:
            group_idx_tgt.append([int(i) for i in contact_groups[name] if 0 <= int(i) < J])
        else:
            group_idx_tgt.append([])
    keep = [c for c, idxs in enumerate(group_idx_tgt) if len(idxs) > 0]
    if len(keep) < len(group_idx_tgt):
        group_idx_tgt = [group_idx_tgt[c] for c in keep]
        sched_tgt = sched_tgt[:, keep]
        group_names = [group_names[c] for c in keep] if group_names else None

    limb_tgt = np.asarray(q_star.get('limb_usage', []), dtype=np.float32)
    limb_tgt_t = torch.tensor(limb_tgt, dtype=torch.float32, device=device) if limb_tgt.size else None
    chains = target_skel_cond.get('kinematic_chains', [])
    K_tgt = len(chains)

    w_sub_t = torch.tensor(_weighted_subtree(parents, offsets_np),
                           dtype=torch.float32, device=device)
    offsets_t = torch.tensor(offsets_np, dtype=torch.float32, device=device)

    # ------ NEW: theta warm-start ------
    if theta_init is not None:
        ti = np.asarray(theta_init, dtype=np.float32)
        out = np.zeros((T, J, 3), dtype=np.float32)
        t_cp = min(T, ti.shape[0])
        j_cp = min(J, ti.shape[1])
        out[:t_cp, :j_cp, :] = ti[:t_cp, :j_cp, :3]
        # Clamp to [-pi, pi] just like the optimiser does after every step.
        out = np.clip(out, -math.pi, math.pi)
        theta = torch.tensor(out, dtype=torch.float32, device=device,
                             requires_grad=True)
    else:
        theta = torch.zeros(T, J, 3, dtype=torch.float32, device=device,
                            requires_grad=True)

    root_init = _init_root_pos_from_q(q_star, body_scale, rest_root_h, T)
    root_pos = torch.tensor(root_init, dtype=torch.float32, device=device,
                            requires_grad=True)

    # Seed root yaw from heading direction (overrides whatever theta_init had
    # for joint-0 yaw — this matches the original solver's behaviour).
    fwd_np = np.zeros((T, 3), dtype=np.float32)
    if com_path.shape[0] >= 2:
        fwd_np[1:] = com_path[1:] - com_path[:-1]
        fwd_np[0] = fwd_np[1]
    fwd_np[:, 1] = 0.0
    mag = np.linalg.norm(fwd_np, axis=-1, keepdims=True)
    ok_mask = mag.squeeze(-1) > _EPS
    if ok_mask.any():
        fwd_np[ok_mask] = fwd_np[ok_mask] / mag[ok_mask]
        yaw = np.arctan2(fwd_np[:, 0], fwd_np[:, 2])
        with torch.no_grad():
            theta[:, 0, 1] = torch.tensor(yaw, dtype=theta.dtype, device=device)

    optim = torch.optim.Adam([theta, root_pos], lr=0.05)
    n_warm = int(0.6 * n_iters)
    best_total = float('inf')
    best_theta = theta.detach().clone()
    best_root = root_pos.detach().clone()
    last_loss_terms: Dict[str, float] = {}
    loss_trace: List[float] = []

    def closure():
        positions, _ = fk_torch(theta, root_pos, offsets_t, parents)
        centroids = torch.einsum('j,tjd->td', w_sub_t, positions)
        com_rec = (centroids - centroids[0:1]) / body_scale
        loss_com = ((com_rec - com_path_t) ** 2).sum(dim=-1).mean()

        loss_contact = torch.zeros((), dtype=positions.dtype, device=device)
        if sched_tgt.numel() > 0 and len(group_idx_tgt) > 0:
            alpha = 10.0 / max(body_scale, _EPS)
            pred = torch.zeros(T, len(group_idx_tgt), dtype=positions.dtype, device=device)
            for c, idxs in enumerate(group_idx_tgt):
                y = positions[:, idxs, 1]
                pred[:, c] = torch.sigmoid(-alpha * y).mean(dim=1)
                mask = (sched_tgt[:, c] >= 0.5).to(positions.dtype)
                if mask.sum() > 0:
                    loss_contact = loss_contact + ((y.mean(dim=1) ** 2) * mask).sum() / (mask.sum() + _EPS)
            loss_contact = loss_contact + ((pred - sched_tgt) ** 2).mean()

        with torch.no_grad():
            fwd_hat = _forward_from_centroids_torch(centroids.detach())
        heading_rec = _heading_velocity_torch(centroids, fwd_hat, fps=float(fps)) / body_scale
        loss_heading = ((heading_rec - heading_tgt) ** 2).mean()

        cad_rec = (_cadence_soft_torch(_soft_contacts_torch(positions, group_idx_tgt, body_scale), float(fps))
                   if len(group_idx_tgt) > 0 else torch.tensor(0.0, device=device))
        loss_cadence = (cad_rec - cadence_tgt) ** 2

        loss_limb = torch.zeros((), dtype=positions.dtype, device=device)
        if limb_tgt_t is not None and limb_tgt_t.numel() > 0 and K_tgt > 0:
            vel = torch.zeros_like(positions)
            vel[1:] = (positions[1:] - positions[:-1]) * fps
            vel[0] = vel[1]
            ke_j = (0.5 * (vel ** 2).sum(dim=-1)).mean(dim=0)
            energy = torch.zeros(K_tgt, dtype=positions.dtype, device=device)
            for k, chain in enumerate(chains):
                idxs = [int(i) for i in chain if 0 <= int(i) < J]
                if idxs:
                    energy[k] = ke_j[idxs].mean()
            energy = energy / (energy.sum() + _EPS)
            K_m = min(K_tgt, limb_tgt_t.numel())
            loss_limb = ((energy[:K_m] - limb_tgt_t[:K_m]) ** 2).mean()

        loss_smooth = (((theta[2:] - 2 * theta[1:-1] + theta[:-2]) ** 2).mean()
                       if T >= 3 else torch.tensor(0.0, device=device))
        loss_root = ((root_pos[:, 1] - rest_root_h) ** 2).mean()

        terms = {
            'com':       w_cfg['com']       * loss_com,
            'contact':   w_cfg['contact']   * loss_contact,
            'heading':   w_cfg['heading']   * loss_heading,
            'cadence':   w_cfg['cadence']   * loss_cadence,
            'limb':      w_cfg['limb']      * loss_limb,
            'smooth':    w_cfg['smooth']    * loss_smooth,
            'root_pose': w_cfg['root_pose'] * loss_root,
        }
        return sum(terms.values()), terms

    t_start = time.time()
    for it in range(n_iters):
        if it == n_warm:
            for g in optim.param_groups:
                g['lr'] = 0.005
        optim.zero_grad()
        total, terms = closure()
        total.backward()
        torch.nn.utils.clip_grad_norm_([theta, root_pos], max_norm=5.0)
        optim.step()
        with torch.no_grad():
            theta.clamp_(-math.pi, math.pi)
        cur = total.item()
        if track_convergence:
            loss_trace.append(cur)
        if cur < best_total:
            best_total = cur
            best_theta = theta.detach().clone()
            best_root = root_pos.detach().clone()
            last_loss_terms = {k: float(v.detach().item()) for k, v in terms.items()}
        if verbose and (it % 50 == 0 or it == n_iters - 1):
            print(f"  iter {it:4d}  total={cur:.6f}  " +
                  " ".join(f"{k}={float(v.detach().item()):.3e}" for k, v in terms.items()))

    with torch.no_grad():
        positions, _ = fk_torch(best_theta, best_root, offsets_t, parents)
    theta_out = best_theta.detach().cpu().numpy()
    root_out = best_root.detach().cpu().numpy()
    positions_out = positions.detach().cpu().numpy()
    q_rec = _reconstruct_quotient(positions_out, parents, offsets_np, group_idx_tgt,
                                  group_names, chains, fps=fps)
    q_rec['body_scale'] = body_scale

    runtime = time.time() - t_start
    if verbose:
        print(f"  runtime: {runtime:.2f}s")
    result = {
        'theta': theta_out,
        'root_pos': root_out,
        'positions': positions_out,
        'final_loss': last_loss_terms,
        'q_reconstructed': q_rec,
        'body_scale': body_scale,
        'runtime_sec': runtime,
    }
    if track_convergence:
        result['loss_trace'] = loss_trace
    return result
