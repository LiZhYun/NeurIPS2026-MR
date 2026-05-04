"""K_multi_init variant: run Stage-2 IK with 3 random theta-initialisations
per pair (seeds 0/1/2), pick the best Q-match, then run Bridge + Stage 3
the same way as the full K pipeline.

Seed semantics:
  * seed 0 : identical to baseline K (zero theta + yaw seed)
  * seed 1 : yaw-seeded theta + N(0, 0.1 rad) Gaussian perturbation on all
             non-yaw channels
  * seed 2 : yaw-seeded theta + N(0, 0.2 rad) Gaussian perturbation on all
             non-yaw channels  (broader to test deeper local-minima basins)

Outputs:
  * per-pair refined motion -> eval/results/k_compare/K_multi_init/
                               pair_<id>_<src>_to_<tgt>.npy
  * aggregated metrics      -> eval/results/k_compare/K_multi_init/metrics.json
  * per-pair 3-init         -> same metrics.json `per_pair[i].multi_init` dict

The Stage-1 Q extraction, Bridge, Stage-3 projection, and metric definitions
are identical to run_k_pipeline_30pairs.py to keep the comparison apples-to-
apples.  The only difference is Stage-2 runs 3x per pair and keeps the best.

Total wall-time budget: ~15 min (30 pairs × ~30s).
"""
from __future__ import annotations
import math
import os
import sys
import json
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / 'eval/results/k_compare/K_multi_init'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Import helpers / modules we reuse unchanged.
from eval.ik_solver import (
    _body_scale_np,
    _compute_rest_root_height,
    _weighted_subtree,
    _init_root_pos_from_q,
    _reconstruct_quotient,
    _q_component_errors,
    _forward_from_centroids_torch,
    _heading_velocity_torch,
    _cadence_soft_torch,
    _soft_contacts_torch,
    fk_torch,
    DEFAULT_WEIGHTS,
    _EPS,
)
from eval.run_k_pipeline_30pairs import (
    load_assets,
    build_q_star,
    q_component_errors,
    contact_f1,
    skating_proxy,
    build_contact_mask_tj,
    _stratum_label,
)

SEED_PERTURB_STD = {0: 0.0, 1: 0.1, 2: 0.2}  # radians


# ========================== Seeded IK solver ==========================

def _solve_ik_seeded(q_star: dict,
                     target_skel_cond: dict,
                     contact_groups: dict,
                     seed: int = 0,
                     n_iters: int = 400,
                     weights: Optional[dict] = None,
                     verbose: bool = False,
                     fps: int = 30,
                     device: Optional[str] = None) -> dict:
    """Same as eval.ik_solver.solve_ik, but adds a ``seed`` kwarg that drives
    a Gaussian perturbation on theta's non-yaw channels at initialisation.
    seed=0 reproduces the baseline solve_ik exactly.

    Returns the same schema as solve_ik: theta, root_pos, positions,
    final_loss, q_reconstructed, body_scale, runtime_sec.
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

    # --- seeded theta init -----------------------------------------------
    rng = np.random.default_rng(int(seed))
    std_perturb = SEED_PERTURB_STD.get(int(seed), 0.0)
    theta_init_np = np.zeros((T, J, 3), dtype=np.float32)

    # Yaw seed from heading
    fwd_np = np.zeros((T, 3), dtype=np.float32)
    if com_path.shape[0] >= 2:
        fwd_np[1:] = com_path[1:] - com_path[:-1]
        fwd_np[0] = fwd_np[1]
    fwd_np[:, 1] = 0.0
    mag = np.linalg.norm(fwd_np, axis=-1, keepdims=True)
    ok = mag.squeeze(-1) > _EPS
    yaw = None
    if ok.any():
        fwd_np[ok] = fwd_np[ok] / mag[ok]
        yaw = np.arctan2(fwd_np[:, 0], fwd_np[:, 2])
        theta_init_np[:, 0, 1] = yaw.astype(np.float32)

    if std_perturb > 0.0:
        perturb = rng.normal(loc=0.0, scale=std_perturb,
                             size=(T, J, 3)).astype(np.float32)
        # Keep the yaw seed on joint 0 channel 1; perturb everything else.
        mask = np.ones((T, J, 3), dtype=np.float32)
        if yaw is not None:
            mask[:, 0, 1] = 0.0
        theta_init_np = theta_init_np + mask * perturb
        # clamp to valid range
        theta_init_np = np.clip(theta_init_np, -math.pi, math.pi)

    theta = torch.tensor(theta_init_np, dtype=torch.float32,
                         device=device, requires_grad=True)
    root_init = _init_root_pos_from_q(q_star, body_scale, rest_root_h, T)
    root_pos = torch.tensor(root_init, dtype=torch.float32, device=device,
                            requires_grad=True)

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
            print(f"  [seed{seed}] iter {it:4d}  total={cur:.6f}")

    with torch.no_grad():
        positions, _ = fk_torch(best_theta, best_root, offsets_t, parents)
    theta_out = best_theta.detach().cpu().numpy()
    root_out = best_root.detach().cpu().numpy()
    positions_out = positions.detach().cpu().numpy()
    q_rec = _reconstruct_quotient(positions_out, parents, offsets_np, group_idx_tgt,
                                  group_names, chains, fps=fps)
    q_rec['body_scale'] = body_scale

    runtime = time.time() - t_start
    return {
        'theta': theta_out,
        'root_pos': root_out,
        'positions': positions_out,
        'final_loss': last_loss_terms,
        'q_reconstructed': q_rec,
        'body_scale': body_scale,
        'runtime_sec': runtime,
        'seed': int(seed),
    }


# ========================= Composite Q-distance =======================

_Q_COMPONENTS = ('com_path', 'heading_vel', 'contact_sched', 'limb_usage')


def _composite_q_distance(q_rec: dict, q_tgt: dict) -> float:
    """Sum of relative-L2 errors across the four Q components + cadence rel.
    Same normalisation as _q_component_errors so it's comparable across pairs.
    NaN-safe: large (1e3) fallback for NaN."""
    errs = _q_component_errors(q_rec, q_tgt)
    total = 0.0
    for k in _Q_COMPONENTS:
        v = errs[k + '_rel_l2']
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            v = 1e3
        total += float(v)
    cad = errs['cadence_rel']
    if cad is None or (isinstance(cad, float) and (math.isnan(cad) or math.isinf(cad))):
        cad = 1e3
    total += float(cad)
    return total


# =============================== Runner ==============================

def run():
    with open(PROJECT_ROOT / 'idea-stage/eval_pairs.json') as f:
        eval_data = json.load(f)
    pairs = eval_data['pairs']

    cond, contact_groups, motion_dir = load_assets()

    from eval.quotient_extractor import extract_quotient
    from eval.k_pipeline_bridge import theta_to_motion_13dim, bridge_diagnostics
    from eval.anytop_projection import anytop_project

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    per_pair = []
    t_total_0 = time.time()
    bridge_diag_agg = []

    for p in pairs:
        pid = int(p['pair_id'])
        src_skel = p['source_skel']
        tgt_skel = p['target_skel']
        src_fname = p['source_fname']
        print(f'\n=== pair {pid:02d}  {src_skel}({src_fname}) -> {tgt_skel}  '
              f'gap={p["family_gap"]}  supp={p["support_same_label"]} ===')

        rec = {
            'pair_id': pid,
            'source_skel': src_skel,
            'target_skel': tgt_skel,
            'source_fname': src_fname,
            'family_gap': p['family_gap'],
            'support_same_label': int(p['support_same_label']),
            'stratum': _stratum_label(p),
            'status': 'pending',
            'error': None,
        }

        t0 = time.time()
        try:
            # --- Stage 1 ---
            q_src = extract_quotient(src_fname, cond[src_skel],
                                     contact_groups=contact_groups,
                                     motion_dir=motion_dir)
            rec['n_frames'] = int(q_src['n_frames'])
            rec['cadence_src'] = float(q_src['cadence'])

            # --- Remap + build Q* ---
            q_star = build_q_star(q_src, src_skel, tgt_skel, contact_groups, cond)

            # --- Stage 2 multi-init ---
            best = None
            cand_dists: List[float] = []
            cand_losses: List[float] = []
            cand_runtimes: List[float] = []
            best_seed = None
            for seed in (0, 1, 2):
                ik_out = _solve_ik_seeded(
                    q_star, cond[tgt_skel], contact_groups[tgt_skel],
                    seed=seed, n_iters=400, verbose=False, device=device)
                q_rec = ik_out['q_reconstructed']
                q_dist = _composite_q_distance(q_rec, q_star)
                cand_dists.append(q_dist)
                cand_losses.append(float(sum(ik_out['final_loss'].values())))
                cand_runtimes.append(float(ik_out['runtime_sec']))
                if best is None or q_dist < best['q_dist']:
                    best = {'ik_out': ik_out, 'q_dist': q_dist}
                    best_seed = seed

            ik_out = best['ik_out']
            rec['multi_init'] = {
                'q_dist_per_seed':     cand_dists,
                'final_loss_per_seed': cand_losses,
                'runtime_per_seed':    cand_runtimes,
                'best_seed':           int(best_seed),
                'q_dist_min':          float(min(cand_dists)),
                'q_dist_max':          float(max(cand_dists)),
                'q_dist_mean':         float(np.mean(cand_dists)),
                'q_dist_spread':       float(max(cand_dists) - min(cand_dists)),
                'q_dist_rel_spread':   float((max(cand_dists) - min(cand_dists))
                                             / (min(cand_dists) + 1e-8)),
            }
            rec['ik_runtime'] = float(sum(cand_runtimes))
            q_rec = ik_out['q_reconstructed']
            errs = q_component_errors(q_rec, q_star)
            rec['q_errors'] = errs
            rec['contact_f1'] = contact_f1(q_rec['contact_sched'],
                                           q_star['contact_sched'])

            # --- Bridge ---
            try:
                motion_13 = theta_to_motion_13dim(
                    ik_out['theta'], ik_out['root_pos'],
                    ik_out['positions'],
                    tgt_skel, cond,
                    contact_groups=contact_groups)
            except Exception as e:
                print(f'  bridge warning: {e}; using identity-6D fallback')
                motion_13 = theta_to_motion_13dim(
                    ik_out['theta'], ik_out['root_pos'],
                    ik_out['positions'],
                    tgt_skel, cond,
                    contact_groups=contact_groups,
                    fit_rotations=False)
            diag = bridge_diagnostics(motion_13, tgt_skel, cond)
            rec['bridge_diag'] = diag
            bridge_diag_agg.append(diag)

            # --- Stage 3 ---
            contact_mask_TJ = build_contact_mask_tj(motion_13)
            com_path_T3 = q_star['com_path'] * q_star['body_scale']
            try:
                proj = anytop_project(
                    motion_13, tgt_skel,
                    hard_constraints={'contact_positions': contact_mask_TJ,
                                      'com_path': com_path_T3},
                    t_init=0.3, n_steps=10, device=device)
                rec['anytop_runtime'] = float(proj['runtime_seconds'])
                out_motion = proj['x_refined']
                rec['status'] = 'ok'
            except Exception as e:
                print(f'  Stage 3 failed: {e}; saving IK-bridge output instead')
                rec['status'] = 'stage3_failed'
                rec['error'] = str(e)
                out_motion = motion_13

            out_path = OUT_DIR / f'pair_{pid:02d}_{src_skel}_to_{tgt_skel}.npy'
            np.save(out_path, out_motion)
            rec['output_path'] = str(out_path)
            rec['skating_proxy'] = skating_proxy(out_motion, contact_groups,
                                                 tgt_skel, cond)
            rec['pair_runtime'] = float(time.time() - t0)
            print(f'  ok: q_errs.com_rel={errs["com_path_rel_l2"]:.3f}  '
                  f'contact_F1={rec["contact_f1"]:.3f}  '
                  f'best_seed={best_seed}  '
                  f'q_dist=[{cand_dists[0]:.3f},{cand_dists[1]:.3f},{cand_dists[2]:.3f}]  '
                  f'pair_runtime={rec["pair_runtime"]:.1f}s')
        except Exception as e:
            rec['status'] = 'failed'
            rec['error'] = str(e) + '\n' + traceback.format_exc(limit=3)
            rec['pair_runtime'] = float(time.time() - t0)
            print(f'  FAILED: {e}')
        per_pair.append(rec)

    total_time = time.time() - t_total_0

    # Stratified means (same schema as run_k_pipeline_30pairs).
    strata_order = ['near_present', 'absent', 'moderate', 'extreme']
    strata = {s: [] for s in strata_order}
    for r in per_pair:
        strata[r['stratum']].append(r)

    def _stratum_stats(entries):
        ok = [e for e in entries if e['status'] == 'ok']
        if not ok:
            return {'n_total': len(entries), 'n_ok': 0}
        qs = lambda k: float(np.mean([e['q_errors'][k] for e in ok]))
        mi = lambda k: float(np.mean([e['multi_init'][k] for e in ok]))
        return {
            'n_total': len(entries),
            'n_ok': len(ok),
            'com_path_rel_l2':     qs('com_path_rel_l2'),
            'heading_vel_rel_l2':  qs('heading_vel_rel_l2'),
            'contact_sched_mae':   qs('contact_sched_mae'),
            'cadence_abs':         qs('cadence_abs'),
            'limb_usage_rel_l2':   qs('limb_usage_rel_l2'),
            'contact_f1':          float(np.mean([e['contact_f1'] for e in ok])),
            'skating_proxy':       float(np.mean([e['skating_proxy'] for e in ok])),
            'ik_runtime':          float(np.mean([e['ik_runtime'] for e in ok])),
            'anytop_runtime':      float(np.mean([e.get('anytop_runtime', 0.0) for e in ok])),
            'pair_runtime':        float(np.mean([e['pair_runtime'] for e in ok])),
            'mi_q_dist_mean':      mi('q_dist_mean'),
            'mi_q_dist_spread':    mi('q_dist_spread'),
            'mi_q_dist_rel_spread': mi('q_dist_rel_spread'),
        }
    strata_stats = {s: _stratum_stats(strata[s]) for s in strata_order}

    # Aggregate multi-init stats.
    ok_pairs = [r for r in per_pair if r['status'] == 'ok']
    mi_agg = {}
    if ok_pairs:
        mi_agg = {
            'mean_q_dist_spread': float(np.mean([r['multi_init']['q_dist_spread']
                                                  for r in ok_pairs])),
            'mean_q_dist_rel_spread': float(np.mean([r['multi_init']['q_dist_rel_spread']
                                                      for r in ok_pairs])),
            'max_q_dist_spread': float(np.max([r['multi_init']['q_dist_spread']
                                                for r in ok_pairs])),
            'max_q_dist_rel_spread': float(np.max([r['multi_init']['q_dist_rel_spread']
                                                    for r in ok_pairs])),
            'best_seed_histogram': {
                str(s): int(sum(1 for r in ok_pairs
                                if r['multi_init']['best_seed'] == s))
                for s in (0, 1, 2)
            },
            'mean_q_dist_per_seed': [
                float(np.mean([r['multi_init']['q_dist_per_seed'][s]
                               for r in ok_pairs]))
                for s in range(3)
            ],
        }

    # Bridge diagnostics aggregate.
    bridge_agg = {}
    if bridge_diag_agg:
        keys = ['ric_norm_median_over_scale', 'ric_norm_p95_over_scale',
                'rot_col0_norm_median', 'vel_mean_abs', 'vel_max_abs',
                'root_y_median_over_scale', 'contact_fraction']
        for k in keys:
            bridge_agg[k] = float(np.mean([d[k] for d in bridge_diag_agg]))

    out = {
        'variant': 'K_multi_init',
        'n_inits_per_pair': 3,
        'seed_perturb_std': SEED_PERTURB_STD,
        'total_runtime_sec': total_time,
        'n_pairs': len(pairs),
        'n_ok':     sum(1 for r in per_pair if r['status'] == 'ok'),
        'n_stage3_failed': sum(1 for r in per_pair if r['status'] == 'stage3_failed'),
        'n_failed': sum(1 for r in per_pair if r['status'] == 'failed'),
        'multi_init_aggregate': mi_agg,
        'strata_stats': strata_stats,
        'bridge_diagnostics_mean': bridge_agg,
        'per_pair': per_pair,
    }
    with open(OUT_DIR / 'metrics.json', 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f'\n=== DONE: total {total_time:.1f}s, n_ok={out["n_ok"]}/{out["n_pairs"]}, '
          f'stage3_failed={out["n_stage3_failed"]}, failed={out["n_failed"]} ===')
    print(f'metrics saved to {OUT_DIR / "metrics.json"}')
    return out


if __name__ == '__main__':
    run()
