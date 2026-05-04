"""Track A: evaluate any retargeting method on synthetic topology-edit pairs.

Inputs: directory of .npz pairs from eval.topology_editing
Each pair contains source (positions, parents, offsets) + target (positions, parents, offsets).

For each pair:
  - Compute the method's output on (source motion, target skeleton)
  - Compute joint-position error vs known target positions
  - Compute contact F1 vs known target contacts
  - Compute analytic ψ-component MSE between method output and known target

Methods to evaluate:
  - Our behavior-conditioned model (when trained)
  - Baselines (text-only, analytic-only, retrieval, null)

Usage:
    conda run -n anytop python -m eval.track_a_eval --method retrieval --out eval/results/track_a_retrieval.json
"""
import os
import json
import argparse
import numpy as np
from os.path import join as pjoin


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pairs_dir', default='eval/results/track_a_synthetic')
    p.add_argument('--method', choices=['retrieval', 'null', 'identity', 'placeholder'], default='retrieval')
    p.add_argument('--out', default='eval/results/track_a_eval.json')
    return p.parse_args()


def joint_position_error(pred_positions, gt_positions, n_joints):
    """MPJPE in meters (or whatever scale data is in). Assumes both are [T, J, 3]."""
    p = pred_positions[:, :n_joints]
    g = gt_positions[:, :n_joints]
    if p.shape[0] != g.shape[0]:
        T = min(p.shape[0], g.shape[0])
        p = p[:T]
        g = g[:T]
    return float(np.sqrt(((p - g) ** 2).sum(axis=-1)).mean())


def contact_f1(pred_positions, gt_positions, n_joints, height_thresh=0.05, vel_thresh=0.05):
    """F1 of contact predictions (joint near ground AND near-zero velocity)."""
    def contact(pos, n):
        pos = pos[:, :n]
        T = pos.shape[0]
        h = pos[:, :, 1]  # y up
        h_low = h < (h.min() + height_thresh)
        v = np.zeros_like(h)
        if T > 1:
            v[1:] = np.sqrt(((pos[1:] - pos[:-1]) ** 2).sum(axis=-1))
        v_low = v < vel_thresh
        return h_low & v_low

    p_c = contact(pred_positions, n_joints)
    g_c = contact(gt_positions, n_joints)
    if p_c.shape != g_c.shape:
        T = min(p_c.shape[0], g_c.shape[0])
        J = min(p_c.shape[1], g_c.shape[1])
        p_c = p_c[:T, :J]
        g_c = g_c[:T, :J]

    tp = (p_c & g_c).sum()
    fp = (p_c & ~g_c).sum()
    fn = (~p_c & g_c).sum()
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    return float(2 * prec * rec / (prec + rec + 1e-8))


def psi_component_mse(pred_positions, gt_positions, parents_pred, offsets_pred,
                       parents_gt, offsets_gt, n_pred, n_gt):
    """Compute MSE between ψ components extracted from pred and gt motions."""
    from eval.effect_program import extract_effect_program
    p = pred_positions[:, :n_pred]
    g = gt_positions[:, :n_gt]
    eff_p = extract_effect_program(p, parents_pred[:n_pred], offsets_pred[:n_pred])
    eff_g = extract_effect_program(g, parents_gt[:n_gt], offsets_gt[:n_gt])
    return {
        'tau_mse': float(((eff_p['tau'] - eff_g['tau']) ** 2).mean()),
        'mu_mse':  float(((eff_p['mu']  - eff_g['mu'])  ** 2).mean()),
        'eta_mse': float(((eff_p['eta'] - eff_g['eta']) ** 2).mean()),
        'rho_mse': float(((eff_p['rho'] - eff_g['rho']) ** 2).mean()),
    }


def predict_retrieval(source_positions, source_parents, source_offsets,
                      target_parents, target_offsets, target_n_joints):
    """Retrieval baseline placeholder: just return source positions padded/truncated to target J.

    For Track A pairs (subdivide/prune/duplicate/merge), there's high overlap, so this gives
    a strong simple baseline.
    """
    T, J, _ = source_positions.shape
    out = np.zeros((T, target_n_joints, 3), dtype=source_positions.dtype)
    common = min(J, target_n_joints)
    out[:, :common] = source_positions[:, :common]
    return out


def predict_null(target_n_joints, T):
    """Null baseline: zero motion."""
    return np.zeros((T, target_n_joints, 3))


def predict_identity_per_op(pair, op):
    """Identity baseline that knows the op type — strongest possible baseline (oracle)."""
    src = pair['source_positions']
    n_pred = pair['target_n_joints'].item() if hasattr(pair['target_n_joints'], 'item') else int(pair['target_n_joints'])
    if op == 'subdivide':
        # Insert midpoint at end (matches subdivide_bone behavior)
        T, J, _ = src.shape
        out = np.zeros((T, J + 1, 3), dtype=src.dtype)
        out[:, :J] = src
        # Midpoint = average of parent (joint 0) and child (joint 2)
        out[:, J] = (src[:, 0] + src[:, 2]) / 2.0
        return out[:, :n_pred]
    elif op == 'prune':
        # First leaf removed — but we don't know which. Just truncate.
        return src[:, :n_pred]
    elif op == 'duplicate':
        # Duplicate first leaf
        T, J, _ = src.shape
        out = np.zeros((T, J + 1, 3), dtype=src.dtype)
        out[:, :J] = src
        out[:, J] = src[:, -1]
        return out[:, :n_pred]
    elif op == 'merge':
        # Skip a degree-2 joint
        return src[:, :n_pred]
    else:
        return src[:, :n_pred]


def main():
    args = parse_args()

    pair_files = sorted(f for f in os.listdir(args.pairs_dir) if f.endswith('.npz'))
    print(f"Evaluating {len(pair_files)} pairs with method={args.method}")

    results = []
    for pf in pair_files:
        data = np.load(pjoin(args.pairs_dir, pf), allow_pickle=True)
        op = str(data['op'])
        source_positions = data['source_positions']
        target_positions = data['target_positions']
        n_pred = data['target_parents'].shape[0]
        n_gt = data['target_parents'].shape[0]

        if args.method == 'retrieval':
            pred = predict_retrieval(source_positions, data['source_parents'], data['source_offsets'],
                                     data['target_parents'], data['target_offsets'], n_pred)
        elif args.method == 'null':
            pred = predict_null(n_pred, source_positions.shape[0])
        elif args.method == 'identity':
            pred = predict_identity_per_op(data, op)
        else:
            print(f"  Method {args.method} not implemented — skipping")
            continue

        # Pad pred if fewer joints
        if pred.shape[1] < n_pred:
            padded = np.zeros((pred.shape[0], n_pred, 3), dtype=pred.dtype)
            padded[:, :pred.shape[1]] = pred
            pred = padded

        mpjpe = joint_position_error(pred, target_positions, n_pred)
        c_f1 = contact_f1(pred, target_positions, n_pred)
        try:
            psi_mse = psi_component_mse(
                pred, target_positions,
                data['target_parents'], data['target_offsets'],
                data['target_parents'], data['target_offsets'],
                n_pred, n_gt)
        except Exception as e:
            psi_mse = {'tau_mse': float('nan'), 'mu_mse': float('nan'),
                       'eta_mse': float('nan'), 'rho_mse': float('nan')}

        results.append({
            'pair': pf,
            'op': op,
            'mpjpe': mpjpe,
            'contact_f1': c_f1,
            **psi_mse,
        })

    # Aggregate by op
    print("\n" + "=" * 50)
    print(f"TRACK A RESULTS — method={args.method}, n_pairs={len(results)}")
    print("=" * 50)

    by_op = {}
    for r in results:
        by_op.setdefault(r['op'], []).append(r)

    for op, rs in sorted(by_op.items()):
        mpjpes = [r['mpjpe'] for r in rs]
        f1s = [r['contact_f1'] for r in rs]
        tau = [r['tau_mse'] for r in rs if not np.isnan(r['tau_mse'])]
        print(f"\n{op}: n={len(rs)}")
        print(f"  MPJPE:      {np.mean(mpjpes):.3f} ± {np.std(mpjpes):.3f}")
        print(f"  Contact F1: {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
        if tau:
            print(f"  τ MSE:      {np.mean(tau):.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'method': args.method, 'results': results}, f, indent=2)
    print(f"\nSaved → {args.out}")


if __name__ == '__main__':
    main()
