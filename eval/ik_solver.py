"""Constrained-IK trajectory-fitting solver (Stage 2 of Idea K).

Given a target skeleton and Q* (from eval.quotient_extractor), solve for
per-frame joint axis-angle rotations theta in R^{T x J x 3} whose FK matches Q*.

Parameterisation : theta[t, j, :3] axis-angle (joint 0 = world-frame root).
FK               : Rodrigues + top-down compose along parents (torch, autograd).
Loss             : weighted COM, contact schedule, heading vel, cadence, limb
                   usage, smoothness, and a root-height anchor.
Optimiser        : Adam; 2 phases (warm lr=0.05 -> refine lr=0.005); grad clip;
                   theta box projection to [-pi, pi].
"""
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

ROOT = Path(str(PROJECT_ROOT))
_EPS = 1e-8


# =============================== FK (torch) ==============================


def _rodrigues(theta: torch.Tensor) -> torch.Tensor:
    """theta [..., 3] -> rotation matrix [..., 3, 3] via Rodrigues' formula."""
    angle = torch.linalg.norm(theta, dim=-1, keepdim=True).clamp_min(_EPS)
    axis = theta / angle
    angle = angle.squeeze(-1)
    ax, ay, az = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = torch.zeros_like(ax)
    K = torch.stack([
        torch.stack([zero, -az, ay], dim=-1),
        torch.stack([az, zero, -ax], dim=-1),
        torch.stack([-ay, ax, zero], dim=-1),
    ], dim=-2)
    I = torch.eye(3, dtype=theta.dtype, device=theta.device).expand(K.shape)
    sin = torch.sin(angle).unsqueeze(-1).unsqueeze(-1)
    cos_m = (1.0 - torch.cos(angle)).unsqueeze(-1).unsqueeze(-1)
    return I + sin * K + cos_m * (K @ K)


def fk_torch(theta: torch.Tensor, root_pos: torch.Tensor,
             offsets: torch.Tensor, parents: List[int]
             ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Differentiable FK. theta:[T,J,3] root_pos:[T,3] offsets:[J,3] parents:list[J]
    Returns positions [T,J,3] and world rotations [T,J,3,3]."""
    T, J, _ = theta.shape
    R_local = _rodrigues(theta)
    R_world = [None] * J
    P_world = [None] * J
    R_world[0] = R_local[:, 0]
    P_world[0] = root_pos + torch.einsum('tij,j->ti', R_local[:, 0], offsets[0])
    for j in range(1, J):
        p = parents[j]
        if p < 0:
            R_world[j] = R_local[:, j]
            P_world[j] = root_pos + torch.einsum('tij,j->ti', R_local[:, j], offsets[j])
            continue
        R_world[j] = R_world[p] @ R_local[:, j]
        P_world[j] = P_world[p] + torch.einsum('tij,j->ti', R_world[p], offsets[j])
    return torch.stack(P_world, dim=1), torch.stack(R_world, dim=1)


# ================================ Helpers ================================

def _subtree_masses_np(parents: List[int], offsets: np.ndarray) -> np.ndarray:
    J = len(parents)
    masses = np.linalg.norm(offsets, axis=1).copy()
    for j in reversed(range(J)):
        p = parents[j]
        if 0 <= p < J and p != j:
            masses[p] += masses[j]
    return masses


def _body_scale_np(offsets: np.ndarray) -> float:
    return float(np.linalg.norm(offsets, axis=1).sum() + _EPS)


def _weighted_subtree(parents, offsets):
    w = _subtree_masses_np(parents, offsets)
    return w / (w.sum() + _EPS)


def _compute_rest_root_height(parents: List[int], offsets: np.ndarray) -> float:
    J = len(parents)
    rest = np.zeros_like(offsets)
    for j in range(1, J):
        p = parents[j]
        if 0 <= p < J:
            rest[j] = rest[p] + offsets[j]
    y_min = rest[:, 1].min()
    return float(-y_min) if y_min < 0 else 0.1


def _forward_from_centroids_torch(centroids: torch.Tensor, smoothing: float = 0.9) -> torch.Tensor:
    T = centroids.shape[0]
    vel = torch.zeros_like(centroids)
    vel[1:] = centroids[1:] - centroids[:-1]
    vel[0] = vel[1]
    vel_h = vel.clone()
    vel_h[:, 1] = 0.0
    forward = torch.zeros_like(centroids)
    running = vel_h[0]
    for t in range(T):
        running = smoothing * running + (1.0 - smoothing) * vel_h[t]
        forward[t] = running
    n = torch.linalg.norm(forward, dim=-1, keepdim=True).clamp_min(_EPS)
    forward = forward / n
    if torch.linalg.norm(forward).item() < _EPS:
        forward[:] = torch.tensor([1.0, 0.0, 0.0], dtype=forward.dtype, device=forward.device)
    return forward


def _heading_velocity_torch(centroids: torch.Tensor, forward: torch.Tensor, fps: float) -> torch.Tensor:
    T = centroids.shape[0]
    vel = torch.zeros_like(centroids)
    vel[1:] = (centroids[1:] - centroids[:-1]) * fps
    vel[0] = vel[1]
    return (vel * forward).sum(dim=-1)


def _cadence_soft_torch(sched: torch.Tensor, fps: float,
                        min_cycle: float = 0.25, max_cycle: float = 4.0) -> torch.Tensor:
    """Power-weighted mean frequency in [1/max_cycle, 1/min_cycle] (differentiable proxy)."""
    s = sched.sum(dim=1) if sched.dim() > 1 else sched
    s = s - s.mean()
    T = s.shape[0]
    if T < 4:
        return torch.tensor(0.0, dtype=s.dtype, device=s.device)
    fft = torch.fft.rfft(s)
    power = fft.real ** 2 + fft.imag ** 2
    freqs = torch.fft.rfftfreq(T, d=1.0 / fps).to(s.device)
    mask = ((freqs >= 1.0 / max_cycle) & (freqs <= 1.0 / min_cycle)).to(power.dtype)
    num = (power * freqs * mask).sum()
    den = (power * mask).sum().clamp_min(_EPS)
    return num / den


def _soft_contacts_torch(positions: torch.Tensor, group_idx: List[List[int]],
                          height_scale: float) -> torch.Tensor:
    T = positions.shape[0]
    C = len(group_idx)
    out = torch.zeros(T, C, dtype=positions.dtype, device=positions.device)
    alpha = 10.0 / max(height_scale, _EPS)
    for c, idxs in enumerate(group_idx):
        if not idxs:
            continue
        y = positions[:, idxs, 1]
        out[:, c] = torch.sigmoid(-alpha * y).mean(dim=1)
    return out


# =============================== Solver core =============================

DEFAULT_WEIGHTS = {
    'com':       20.0,
    'contact':    1.0,
    'smooth':     0.02,
    'heading':    5.0,
    'cadence':    0.05,
    'limb':       0.5,
    'root_pose':  0.1,
}


def _init_root_pos_from_q(q_star, body_scale, rest_root_h, T_target):
    com = np.asarray(q_star['com_path'], dtype=np.float32)
    if com.shape[0] != T_target:
        idx = np.linspace(0, com.shape[0] - 1, T_target)
        com = np.stack([np.interp(idx, np.arange(com.shape[0]), com[:, d]) for d in range(3)], -1)
    root = com * body_scale
    root[:, 1] += rest_root_h
    return root


def solve_ik(q_star: dict,
             target_skel_cond: dict,
             contact_groups: dict,
             n_iters: int = 400,
             weights: Optional[dict] = None,
             verbose: bool = False,
             fps: int = 30,
             device: Optional[str] = None) -> dict:
    """See module docstring. Returns theta, root_pos, positions, final_loss, q_reconstructed."""
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
    heading_tgt = torch.tensor(np.asarray(q_star['heading_vel'], dtype=np.float32), device=device)
    cadence_tgt = float(q_star.get('cadence', 0.0))
    sched_tgt = torch.tensor(np.asarray(q_star['contact_sched'], dtype=np.float32), device=device)
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

    w_sub_t = torch.tensor(_weighted_subtree(parents, offsets_np), dtype=torch.float32, device=device)
    offsets_t = torch.tensor(offsets_np, dtype=torch.float32, device=device)

    theta = torch.zeros(T, J, 3, dtype=torch.float32, device=device, requires_grad=True)
    root_init = _init_root_pos_from_q(q_star, body_scale, rest_root_h, T)
    root_pos = torch.tensor(root_init, dtype=torch.float32, device=device, requires_grad=True)

    # Seed root yaw from heading direction
    fwd_np = np.zeros((T, 3), dtype=np.float32)
    if com_path.shape[0] >= 2:
        fwd_np[1:] = com_path[1:] - com_path[:-1]
        fwd_np[0] = fwd_np[1]
    fwd_np[:, 1] = 0.0
    mag = np.linalg.norm(fwd_np, axis=-1, keepdims=True)
    ok = mag.squeeze(-1) > _EPS
    if ok.any():
        fwd_np[ok] = fwd_np[ok] / mag[ok]
        yaw = np.arctan2(fwd_np[:, 0], fwd_np[:, 2])
        with torch.no_grad():
            theta[:, 0, 1] = torch.tensor(yaw, dtype=theta.dtype, device=device)

    optim = torch.optim.Adam([theta, root_pos], lr=0.05)
    n_warm = int(0.6 * n_iters)
    best_total = float('inf')
    best_theta = theta.detach().clone()
    best_root = root_pos.detach().clone()
    last_loss_terms: Dict[str, float] = {}

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
    return {
        'theta': theta_out,
        'root_pos': root_out,
        'positions': positions_out,
        'final_loss': last_loss_terms,
        'q_reconstructed': q_rec,
        'body_scale': body_scale,
        'runtime_sec': runtime,
    }


# ====================== Q reconstruction + errors ========================

def _reconstruct_quotient(positions: np.ndarray, parents: List[int], offsets: np.ndarray,
                          group_idx: List[List[int]], group_names: Optional[List[str]],
                          chains: List[List[int]], fps: int = 30) -> dict:
    from eval.effect_program import canonical_body_frame
    from eval.quotient_extractor import compute_heading_velocity, compute_cadence, compute_limb_usage

    subtree = _subtree_masses_np(parents, offsets)
    centroids, R = canonical_body_frame(positions, subtree)
    scale = _body_scale_np(offsets)
    com_path = (centroids - centroids[0:1]) / scale
    heading_vel = compute_heading_velocity(centroids, R, fps=fps) / scale

    T, J, _ = positions.shape
    y = positions[..., 1]
    ground = np.percentile(y, 5)
    tau = max(0.02 * scale, 1e-3)
    contacts = (y - ground < tau).astype(np.int8)
    if group_idx and group_names:
        C = len(group_idx)
        sched = np.zeros((T, C), dtype=np.float32)
        for c, idxs in enumerate(group_idx):
            if idxs:
                sched[:, c] = contacts[:, idxs].mean(axis=1)
    else:
        sched = contacts.sum(axis=1).astype(np.float32) / max(J, 1)

    cadence = compute_cadence(sched, fps=fps)
    limb_usage = compute_limb_usage(positions, chains, fps=fps)

    return {
        'com_path': com_path.astype(np.float32),
        'heading_vel': heading_vel.astype(np.float32),
        'contact_sched': sched.astype(np.float32),
        'contact_group_names': group_names,
        'cadence': float(cadence),
        'limb_usage': limb_usage.astype(np.float32),
        'n_joints': int(J),
        'n_frames': int(T),
    }


def _q_component_errors(q_rec: dict, q_tgt: dict) -> dict:
    """Per-component errors: relative L2, MAE, and range-normed MAE."""
    def err(a, b):
        a = np.asarray(a, dtype=np.float64).ravel()
        b = np.asarray(b, dtype=np.float64).ravel()
        n = min(a.size, b.size)
        if n == 0:
            return float('nan'), float('nan'), float('nan')
        a, b = a[:n], b[:n]
        rel = float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-6))
        mae = float(np.abs(a - b).mean())
        rng = float(np.abs(b).max() - np.abs(b).min() + 1e-6)
        return rel, mae, mae / rng
    out = {}
    for k in ('com_path', 'heading_vel', 'contact_sched', 'limb_usage'):
        rel, mae, rng = err(q_rec[k], q_tgt[k])
        out[k + '_rel_l2'] = rel
        out[k + '_mae'] = mae
        out[k + '_rng_mae'] = rng
    out['cadence_abs'] = abs(float(q_rec['cadence']) - float(q_tgt['cadence']))
    out['cadence_rel'] = out['cadence_abs'] / (abs(float(q_tgt['cadence'])) + 1e-6)
    return out


# ================================ Tests ==================================

def _load_cond():
    return np.load(ROOT / 'dataset/truebones/zoo/truebones_processed/cond.npy',
                   allow_pickle=True).item()


def _load_contact_groups():
    with open(ROOT / 'eval/quotient_assets/contact_groups.json') as f:
        return json.load(f)


def test_round_trip(horse_clip='Horse___LandRun_448.npy', n_iters: int = 500):
    print(f"\n=== Round-trip test: {horse_clip} ===")
    from eval.quotient_extractor import extract_quotient
    cond = _load_cond()
    contact_groups = _load_contact_groups()
    q = extract_quotient(horse_clip, cond['Horse'], contact_groups=contact_groups)
    print(f"  Q: T={q['n_frames']}, J={q['n_joints']}, groups={q['contact_group_names']}, "
          f"cadence={q['cadence']:.3f}")
    out = solve_ik(q, cond['Horse'], contact_groups['Horse'], n_iters=n_iters, verbose=True)
    errs = _q_component_errors(out['q_reconstructed'], q)
    print(f"  Round-trip Q errors:")
    for k, v in errs.items():
        print(f"    {k}: {v:.4f}")
    print(f"  runtime: {out['runtime_sec']:.2f}s")
    return out, errs, q


def test_cross_skel(horse_clip='Horse___LandRun_448.npy', target='Cat', n_iters: int = 500):
    print(f"\n=== Cross-skel test: {horse_clip} -> {target} ===")
    from eval.quotient_extractor import extract_quotient
    cond = _load_cond()
    contact_groups = _load_contact_groups()
    q = extract_quotient(horse_clip, cond['Horse'], contact_groups=contact_groups)
    if target not in cond or target not in contact_groups:
        print(f"  SKIP: missing cond or contact groups for {target}")
        return None, None, q
    src_names = q['contact_group_names']
    tgt_groups = contact_groups[target]
    common = [n for n in src_names if n in tgt_groups]
    if not common:
        print(f"  SKIP: no shared groups src={src_names} tgt={list(tgt_groups.keys())}")
        return None, None, q
    col_idx = [src_names.index(n) for n in common]
    q_star = {**q,
              'contact_sched': q['contact_sched'][:, col_idx],
              'contact_group_names': common}
    out = solve_ik(q_star, cond[target], tgt_groups, n_iters=n_iters, verbose=True)
    errs = _q_component_errors(out['q_reconstructed'], q_star)
    print(f"  Cross-skel Q errors vs Q*:")
    for k, v in errs.items():
        print(f"    {k}: {v:.4f}")
    qr = out['q_reconstructed']
    print(f"  Cross-skel sanity: cadence={qr['cadence']:.3f} "
          f"contact_mean={qr['contact_sched'].mean():.3f} "
          f"COM|max|={np.abs(qr['com_path']).max():.3f}")
    print(f"  runtime: {out['runtime_sec']:.2f}s")
    return out, errs, q_star


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--iters', type=int, default=500)
    p.add_argument('--test', type=str, default='both', choices=['round', 'cross', 'both'])
    p.add_argument('--horse_clip', type=str, default='Horse___LandRun_448.npy')
    p.add_argument('--target', type=str, default='Cat')
    args = p.parse_args()
    if args.test in ('round', 'both'):
        test_round_trip(horse_clip=args.horse_clip, n_iters=args.iters)
    if args.test in ('cross', 'both'):
        test_cross_skel(horse_clip=args.horse_clip, target=args.target, n_iters=args.iters)
