"""Run the Idea-N (K_ikprior) pipeline on all 30 canonical eval pairs.

Idea N fuses Stage-2 (IK) and Stage-3 (AnyTop SDEdit-projection) of Idea K
into a single Adam optimisation.  The frozen AnyTop score function acts as
a soft regulariser on the IK.  After the solver converges, we apply the
usual bridge to 13-dim (reusing k_pipeline_bridge) so the output has the
same representation as K / K_no_stage3 / baselines.

Outputs:
  * per-pair motion -> eval/results/k_compare/K_ikprior/pair_<id>_<src>_to_<tgt>.npy
  * metrics.json    -> eval/results/k_compare/K_ikprior/metrics.json
"""
from __future__ import annotations
import os
import sys
import json
import time
import traceback
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / 'eval/results/k_compare/K_ikprior'
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =================== helpers ===================

def load_assets():
    from data_loaders.truebones.truebones_utils.param_utils import DATASET_DIR
    cond = np.load(os.path.join(DATASET_DIR, 'cond.npy'),
                   allow_pickle=True).item()
    with open(PROJECT_ROOT / 'eval/quotient_assets/contact_groups.json') as f:
        cg = json.load(f)
    motion_dir = os.path.join(DATASET_DIR, 'motions')
    return cond, cg, motion_dir


def remap_contact_sched(q_src: dict, src_skel: str, tgt_skel: str,
                        contact_groups: dict) -> tuple:
    src_sched = q_src['contact_sched']
    src_names = q_src.get('contact_group_names') or []
    T = src_sched.shape[0]
    if tgt_skel not in contact_groups:
        return src_sched.copy(), src_names
    tgt_groups = contact_groups[tgt_skel]
    tgt_names = sorted(tgt_groups.keys())
    C_tgt = len(tgt_names)
    if src_sched.ndim == 1:
        broadcast = np.tile(src_sched[:, None], (1, C_tgt))
        return broadcast.astype(np.float32), tgt_names
    overlap = [n for n in tgt_names if n in src_names]
    if overlap:
        sched = np.zeros((T, C_tgt), dtype=np.float32)
        for i, tn in enumerate(tgt_names):
            if tn in src_names:
                sched[:, i] = src_sched[:, src_names.index(tn)]
            else:
                sched[:, i] = src_sched.mean(axis=1)
        return sched, tgt_names
    agg = src_sched.mean(axis=1) if src_sched.ndim == 2 else src_sched
    sched = np.tile(agg[:, None], (1, C_tgt)).astype(np.float32)
    return sched, tgt_names


def build_q_star(q_src: dict, src_skel: str, tgt_skel: str,
                 contact_groups: dict, cond: dict) -> dict:
    sched_tgt, tgt_group_names = remap_contact_sched(
        q_src, src_skel, tgt_skel, contact_groups)
    limb_src = np.asarray(q_src['limb_usage'], dtype=np.float32)
    K_tgt = len(cond[tgt_skel].get('kinematic_chains', []))
    if K_tgt == 0:
        limb_tgt = limb_src.copy()
    elif K_tgt == limb_src.size:
        limb_tgt = limb_src.copy()
    else:
        x_src = np.linspace(0, 1, num=limb_src.size, endpoint=True)
        x_tgt = np.linspace(0, 1, num=K_tgt, endpoint=True)
        limb_tgt = np.interp(x_tgt, x_src, limb_src).astype(np.float32)
        limb_tgt = limb_tgt / (limb_tgt.sum() + 1e-8)
    return {
        'com_path':            q_src['com_path'].astype(np.float32),
        'heading_vel':         q_src['heading_vel'].astype(np.float32),
        'contact_sched':       sched_tgt,
        'contact_group_names': tgt_group_names,
        'cadence':             float(q_src['cadence']),
        'limb_usage':          limb_tgt,
        'body_scale':          float(q_src.get('body_scale', 1.0)),
        'n_joints':            int(cond[tgt_skel]['offsets'].shape[0]),
        'n_frames':            int(q_src['com_path'].shape[0]),
    }


def q_component_errors(q_rec: dict, q_tgt: dict) -> dict:
    from eval.ik_solver import _q_component_errors
    return _q_component_errors(q_rec, q_tgt)


def contact_f1(sched_rec, sched_tgt, thresh=0.5) -> float:
    pred = (np.asarray(sched_rec) >= thresh).astype(np.int8).ravel()
    gt = (np.asarray(sched_tgt) >= thresh).astype(np.int8).ravel()
    n = min(pred.size, gt.size)
    pred, gt = pred[:n], gt[:n]
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)
    return float(2 * p * r / (p + r + 1e-8))


def skating_proxy(motion_13, contact_groups, tgt_skel, cond, fps=30,
                  contact_thresh=0.5) -> float:
    from data_loaders.truebones.truebones_utils.motion_process import (
        recover_from_bvh_ric_np,
    )
    pos = recover_from_bvh_ric_np(motion_13.astype(np.float32))
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
    y_vel = np.zeros((T, len(foot_joints)))
    x_vel = np.zeros_like(y_vel)
    z_vel = np.zeros_like(y_vel)
    if T >= 2:
        dp = pos[1:, foot_joints, :] - pos[:-1, foot_joints, :]
        x_vel[1:] = dp[..., 0]
        y_vel[1:] = dp[..., 1]
        z_vel[1:] = dp[..., 2]
    horiz_speed = np.sqrt(x_vel ** 2 + z_vel ** 2) * fps
    in_contact = (contact[:, foot_joints] > contact_thresh)
    if not in_contact.any():
        return 0.0
    return float(horiz_speed[in_contact].mean())


# =================== main runner ===================

def _stratum_label(p) -> str:
    if p['support_same_label'] == 0:
        return 'absent'
    if p['family_gap'] == 'near':
        return 'near_present'
    if p['family_gap'] == 'moderate':
        return 'moderate'
    if p['family_gap'] == 'extreme':
        return 'extreme'
    return 'other'


def run(w_prior: float = 0.3, prior_every: int = 25, t_fix: int = 30,
        n_iters: int = 400):
    with open(PROJECT_ROOT / 'idea-stage/eval_pairs.json') as f:
        eval_data = json.load(f)
    pairs = eval_data['pairs']

    cond, contact_groups, motion_dir = load_assets()

    from eval.quotient_extractor import extract_quotient
    from eval.ik_solver_n import solve_ik_n
    from eval.k_pipeline_bridge import theta_to_motion_13dim, bridge_diagnostics

    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    per_pair = []
    t_total_0 = time.time()
    bridge_diag_agg = []

    for p in pairs:
        pid = int(p['pair_id'])
        src_skel = p['source_skel']
        tgt_skel = p['target_skel']
        src_fname = p['source_fname']
        print(f'\n=== pair {pid:02d}  {src_skel}({src_fname}) -> {tgt_skel}  gap={p["family_gap"]}  supp={p["support_same_label"]} ===')

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
            # Stage 1
            q_src = extract_quotient(src_fname, cond[src_skel],
                                     contact_groups=contact_groups,
                                     motion_dir=motion_dir)
            rec['n_frames'] = int(q_src['n_frames'])
            rec['cadence_src'] = float(q_src['cadence'])

            q_star = build_q_star(q_src, src_skel, tgt_skel, contact_groups, cond)

            # Stage 2 + Prior (Idea N)
            t2 = time.time()
            ik_out = solve_ik_n(q_star, tgt_skel, cond[tgt_skel],
                                contact_groups[tgt_skel],
                                n_iters=n_iters, verbose=False,
                                device=device,
                                prior_every=prior_every,
                                t_fix=t_fix,
                                weights={'prior': w_prior})
            rec['ik_runtime'] = float(ik_out['runtime_sec'])
            rec['prior_runtime'] = float(ik_out.get('prior_runtime_sec', 0.0))
            rec['n_prior_evals'] = int(ik_out.get('n_prior_evals', 0))
            rec['variant'] = ik_out.get('variant', 'joint_gradient')
            if ik_out.get('prior_history'):
                rec['prior_first'] = float(ik_out['prior_history'][0][1])
                rec['prior_last'] = float(ik_out['prior_history'][-1][1])
            q_rec = ik_out['q_reconstructed']
            errs = q_component_errors(q_rec, q_star)
            rec['q_errors'] = errs
            rec['contact_f1'] = contact_f1(q_rec['contact_sched'],
                                           q_star['contact_sched'])

            # Bridge to 13-dim so the saved motion is compatible with the
            # unified classifier / skating metrics (same as K).
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

            out_path = OUT_DIR / f'pair_{pid:02d}_{src_skel}_to_{tgt_skel}.npy'
            np.save(out_path, motion_13)
            rec['output_path'] = str(out_path)

            rec['skating_proxy'] = skating_proxy(motion_13, contact_groups,
                                                 tgt_skel, cond)
            rec['pair_runtime'] = float(time.time() - t0)
            rec['status'] = 'ok'
            print(f'  ok: q_errs.com_rel={errs["com_path_rel_l2"]:.3f}  '
                  f'contact_F1={rec["contact_f1"]:.3f}  skate={rec["skating_proxy"]:.4f}  '
                  f'prior_0={rec.get("prior_first", 0):.1f}->last={rec.get("prior_last", 0):.1f}  '
                  f'pair_runtime={rec["pair_runtime"]:.1f}s')
        except Exception as e:
            rec['status'] = 'failed'
            rec['error'] = str(e) + '\n' + traceback.format_exc(limit=3)
            rec['pair_runtime'] = float(time.time() - t0)
            print(f'  FAILED: {e}')
        per_pair.append(rec)

    total_time = time.time() - t_total_0

    # stratified means
    strata_order = ['near_present', 'absent', 'moderate', 'extreme']
    strata = {s: [] for s in strata_order}
    for r in per_pair:
        strata[r['stratum']].append(r)

    def _stratum_stats(entries):
        ok = [e for e in entries if e['status'] == 'ok']
        if not ok:
            return {'n_total': len(entries), 'n_ok': 0}
        qs = lambda k: float(np.mean([e['q_errors'][k] for e in ok]))
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
            'prior_runtime':       float(np.mean([e.get('prior_runtime', 0.0) for e in ok])),
            'n_prior_evals':       float(np.mean([e.get('n_prior_evals', 0.0) for e in ok])),
            'prior_first':         float(np.mean([e.get('prior_first', 0.0) for e in ok])),
            'prior_last':          float(np.mean([e.get('prior_last', 0.0) for e in ok])),
            'pair_runtime':        float(np.mean([e['pair_runtime'] for e in ok])),
        }
    strata_stats = {s: _stratum_stats(strata[s]) for s in strata_order}

    # Overall skating + task errors (for quick inspection)
    ok_entries = [r for r in per_pair if r['status'] == 'ok']
    overall = {}
    if ok_entries:
        overall = {
            'com_path_rel_l2': float(np.mean([e['q_errors']['com_path_rel_l2']
                                              for e in ok_entries])),
            'heading_vel_rel_l2': float(np.mean([e['q_errors']['heading_vel_rel_l2']
                                                 for e in ok_entries])),
            'contact_f1':   float(np.mean([e['contact_f1'] for e in ok_entries])),
            'skating_proxy':float(np.mean([e['skating_proxy'] for e in ok_entries])),
            'pair_runtime': float(np.mean([e['pair_runtime'] for e in ok_entries])),
        }

    bridge_agg = {}
    if bridge_diag_agg:
        keys = ['ric_norm_median_over_scale', 'ric_norm_p95_over_scale',
                'rot_col0_norm_median', 'vel_mean_abs', 'vel_max_abs',
                'root_y_median_over_scale', 'contact_fraction']
        for k in keys:
            bridge_agg[k] = float(np.mean([d[k] for d in bridge_diag_agg]))

    out = {
        'variant':           'Idea N (joint gradient IK + prior)',
        'w_prior':            float(w_prior),
        'prior_every':        int(prior_every),
        't_fix':              int(t_fix),
        'n_iters':            int(n_iters),
        'total_runtime_sec':  total_time,
        'n_pairs':            len(pairs),
        'n_ok':               sum(1 for r in per_pair if r['status'] == 'ok'),
        'n_failed':           sum(1 for r in per_pair if r['status'] == 'failed'),
        'overall':            overall,
        'strata_stats':       strata_stats,
        'bridge_diagnostics_mean': bridge_agg,
        'per_pair':           per_pair,
    }
    with open(OUT_DIR / 'metrics.json', 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f'\n=== DONE: total {total_time:.1f}s, n_ok={out["n_ok"]}/{out["n_pairs"]}, '
          f'failed={out["n_failed"]} ===')
    print(f'metrics saved to {OUT_DIR / "metrics.json"}')
    return out


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--w_prior', type=float, default=0.3)
    parser.add_argument('--prior_every', type=int, default=25)
    parser.add_argument('--t_fix', type=int, default=30)
    parser.add_argument('--n_iters', type=int, default=400)
    args = parser.parse_args()
    run(w_prior=args.w_prior, prior_every=args.prior_every, t_fix=args.t_fix,
        n_iters=args.n_iters)
