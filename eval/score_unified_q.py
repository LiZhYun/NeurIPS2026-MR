"""Classifier-INDEPENDENT unified Q comparison across ALL methods.

For each method folder under eval/results/k_compare/, load every
pair_<id>_<src>_to_<tgt>.npy, extract Q on the TARGET skeleton, compare
component-wise with the SOURCE clip's Q.

Outputs:
  idea-stage/unified_q_comparison.json with:
    - per_method_per_pair
    - stratified_means (per method, per stratum, 6 Q + 2 plausibility)
    - rankings (per stratum, rank methods on each metric)
    - pareto (per stratum, Pareto-optimal methods on (Q-pres, plausibility))

Closes the classifier-dependence gap in ROUND2_ABLATIONS_RESULTS.md.
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

ROOT = Path(str(PROJECT_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

K_COMPARE_DIR = ROOT / 'eval/results/k_compare'
EVAL_PAIRS = ROOT / 'idea-stage/eval_pairs.json'
COND_PATH = ROOT / 'dataset/truebones/zoo/truebones_processed/cond.npy'
MOTIONS_DIR = ROOT / 'dataset/truebones/zoo/truebones_processed/motions'
CONTACT_GROUPS_PATH = ROOT / 'eval/quotient_assets/contact_groups.json'
OUT_PATH = ROOT / 'idea-stage/unified_q_comparison.json'

FPS = 30
SKATING_VEL_THRESHOLD = 0.03  # horizontal velocity during contact > this counts as skating
CONTACT_THRESHOLD = 0.5

# Directories (relative to K_COMPARE_DIR) to exclude from scoring.
EXCLUDE_DIRS = {'renders'}


def stratum_of(family_gap: str, support: int) -> list:
    """Return list of strata a pair belongs to."""
    strata = []
    if family_gap in ('near', 'near_present'):
        strata.append('near_present')
    elif family_gap == 'moderate':
        strata.append('moderate')
    elif family_gap == 'extreme':
        strata.append('extreme')
    if support == 0:
        strata.append('absent')
    strata.append('all')
    return strata


def discover_methods() -> list:
    methods = []
    for entry in sorted(K_COMPARE_DIR.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name in EXCLUDE_DIRS:
            continue
        has_pairs = any(entry.glob('pair_*.npy'))
        if has_pairs:
            methods.append(entry.name)
    return methods


def discover_method_pairs(method_dir: Path) -> list:
    """Return list of (pair_id, fname) tuples."""
    pairs = []
    for p in sorted(method_dir.glob('pair_*.npy')):
        stem = p.stem  # pair_00_Src_to_Tgt
        parts = stem.split('_')
        if len(parts) < 4 or parts[0] != 'pair':
            continue
        try:
            pid = int(parts[1])
        except ValueError:
            continue
        pairs.append((pid, p.name))
    return pairs


def compute_source_q(src_fname, cond_dict, contact_groups, extract_quotient):
    """Extract Q for a source fname from the official motion directory."""
    src_skel = src_fname.split('___')[0]
    if src_skel not in cond_dict:
        raise RuntimeError(f'missing cond for source skel {src_skel}')
    return extract_quotient(src_fname, cond_dict[src_skel],
                            contact_groups=contact_groups,
                            motion_dir=str(MOTIONS_DIR))


def compute_target_q_from_npy(npy_path: Path, tgt_skel: str,
                              cond_dict, contact_groups,
                              extract_quotient) -> dict:
    """Extract Q on the target skeleton by temporarily copying the npy
    into the motion directory so extract_quotient can load it."""
    tmp_name = f'__uniq_tmp_{os.getpid()}_{int(time.time()*1e6)}_{npy_path.stem}.npy'
    tmp_path = MOTIONS_DIR / tmp_name
    try:
        x = np.load(npy_path)
        np.save(tmp_path, x.astype(np.float32))
        q = extract_quotient(tmp_name, cond_dict[tgt_skel],
                             contact_groups=contact_groups,
                             motion_dir=str(MOTIONS_DIR))
        return q
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def q_component_distances(q_src: dict, q_tgt: dict) -> dict:
    """Compute the 5 Q-component distances.  Relative L2 = ||a - b|| / (||a|| + 1e-6) for
    com_path / heading_vel; absolute for cadence; aggregate-over-groups L2 for contacts;
    and top-5 limb-usage L2 (padded)."""

    def _resample_time(a: np.ndarray, T: int) -> np.ndarray:
        """Linearly-indexed resample of a[T_a, ...] to T frames."""
        a = np.asarray(a)
        if a.ndim == 0:
            return a
        T_a = a.shape[0]
        if T_a == T:
            return a
        idx = np.clip(np.round(np.linspace(0, T_a - 1, T)).astype(int), 0, T_a - 1)
        return a[idx]

    def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a).reshape(-1).astype(np.float64)
        b = np.asarray(b).reshape(-1).astype(np.float64)
        denom = np.linalg.norm(a) + np.linalg.norm(b) + 1e-6
        return float(np.linalg.norm(a - b) / denom)

    # com_path: [T, 3]
    t_a = q_src['com_path'].shape[0]
    t_b = q_tgt['com_path'].shape[0]
    T = min(t_a, t_b)
    com_a = _resample_time(q_src['com_path'], T)
    com_b = _resample_time(q_tgt['com_path'], T)
    out_com_path = _rel_l2(com_a, com_b)

    # heading_vel: [T]
    t_a = q_src['heading_vel'].shape[0]
    t_b = q_tgt['heading_vel'].shape[0]
    T = min(t_a, t_b)
    h_a = _resample_time(q_src['heading_vel'], T)
    h_b = _resample_time(q_tgt['heading_vel'], T)
    out_heading_vel = _rel_l2(h_a, h_b)

    # cadence scalar
    out_cadence = float(abs(float(q_src['cadence']) - float(q_tgt['cadence'])))

    # contact schedule aggregate
    cs_src = np.asarray(q_src['contact_sched'])
    cs_tgt = np.asarray(q_tgt['contact_sched'])
    agg_src = cs_src.sum(axis=1) if cs_src.ndim == 2 else cs_src
    agg_tgt = cs_tgt.sum(axis=1) if cs_tgt.ndim == 2 else cs_tgt
    T = min(len(agg_src), len(agg_tgt))
    agg_src_r = _resample_time(agg_src, T)
    agg_tgt_r = _resample_time(agg_tgt, T)
    out_contact_sched = _rel_l2(agg_src_r, agg_tgt_r)

    # limb usage top-5
    lu_src = -np.sort(-np.asarray(q_src['limb_usage']))[:5]
    lu_tgt = -np.sort(-np.asarray(q_tgt['limb_usage']))[:5]
    K = max(len(lu_src), len(lu_tgt), 5)
    lu_src = np.pad(lu_src, (0, K - len(lu_src)))
    lu_tgt = np.pad(lu_tgt, (0, K - len(lu_tgt)))
    out_limb_usage = _rel_l2(lu_src, lu_tgt)

    # contact F1 on binarised summed-over-groups schedule
    bin_src = (agg_src_r >= CONTACT_THRESHOLD).astype(np.int8)
    bin_tgt = (agg_tgt_r >= CONTACT_THRESHOLD).astype(np.int8)
    tp = int(((bin_tgt == 1) & (bin_src == 1)).sum())
    fp = int(((bin_tgt == 1) & (bin_src == 0)).sum())
    fn = int(((bin_tgt == 0) & (bin_src == 1)).sum())
    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)

    return {
        'q_com_path_l2': out_com_path,
        'q_heading_vel_l2': out_heading_vel,
        'q_cadence_abs_diff': out_cadence,
        'q_contact_sched_aggregate_l2': out_contact_sched,
        'q_limb_usage_top5_l2': out_limb_usage,
        'contact_f1_vs_source_aggregate': float(f1),
    }


def compute_plausibility(motion: np.ndarray, cond: dict, fps=FPS) -> dict:
    """Compute plausibility metrics that don't require a source:
    - foot skating proxy: mean horizontal-plane speed during contact frames
      (contact threshold 0.5 on channel 12)
    - acceleration smoothness: RMS of 2nd-time-derivative of joint positions.
    Motion shape: [T, J, 13]. Positions reconstructed via recover_from_bvh_ric_np.
    """
    from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np

    J_cond = cond['offsets'].shape[0]
    m = motion.copy()
    if m.ndim != 3 or m.shape[-1] < 13:
        return {'skating_proxy': None, 'accel_rms': None}

    if m.shape[1] > J_cond:
        m = m[:, :J_cond]

    contacts = (m[..., 12] > CONTACT_THRESHOLD).astype(np.int8)  # [T, J]
    try:
        pos = recover_from_bvh_ric_np(m.astype(np.float32))  # [T, J, 3]
    except Exception:
        return {'skating_proxy': None, 'accel_rms': None}

    # Horizontal velocity: X-Z plane (index 0, 2); Y is up
    vel = np.zeros_like(pos)
    vel[1:] = (pos[1:] - pos[:-1]) * fps
    vel[0] = vel[1]
    horiz_speed = np.sqrt(vel[..., 0] ** 2 + vel[..., 2] ** 2)  # [T, J]

    # Body-scale normalise
    body_scale = float(np.linalg.norm(cond['offsets'], axis=1).sum() + 1e-6)
    horiz_speed = horiz_speed / body_scale

    # Skating: mean horizontal speed on contact frames
    mask = contacts.astype(bool)
    skating_vals = horiz_speed[mask]
    skating_proxy = float(skating_vals.mean()) if skating_vals.size > 0 else 0.0

    # Accel RMS (positions second derivative), body-scale normalised
    acc = np.zeros_like(pos)
    acc[1:-1] = (pos[2:] - 2 * pos[1:-1] + pos[:-2]) * (fps ** 2)
    acc_mag = np.linalg.norm(acc, axis=-1) / body_scale  # [T, J]
    accel_rms = float(np.sqrt((acc_mag[1:-1] ** 2).mean()))

    return {'skating_proxy': skating_proxy, 'accel_rms': accel_rms}


def stratified_means(per_pair_entries: list, strata_keys: list,
                     metric_keys: list) -> dict:
    """For each stratum, for each metric, compute mean over pairs in that stratum."""
    buckets = defaultdict(list)
    for e in per_pair_entries:
        for s in e.get('strata', []):
            buckets[s].append(e)
    out = {}
    for s in strata_keys:
        es = buckets.get(s, [])
        stats = {'n': len(es)}
        for k in metric_keys:
            vals = [e[k] for e in es if e.get(k) is not None
                    and not (isinstance(e[k], float) and np.isnan(e[k]))]
            stats[k] = float(np.mean(vals)) if vals else None
        out[s] = stats
    return out


def compute_rankings(stratified_per_method: dict, strata_keys: list,
                     metric_keys: list, higher_is_better: set) -> dict:
    """For each stratum and metric, rank methods 1..N. 1 is best."""
    rankings = {}
    methods = list(stratified_per_method.keys())
    for s in strata_keys:
        rankings[s] = {}
        for k in metric_keys:
            vals = []
            for m in methods:
                v = stratified_per_method[m].get(s, {}).get(k)
                vals.append((m, v))
            # Separate None from not-None
            valid = [(m, v) for m, v in vals if v is not None]
            invalid = [m for m, v in vals if v is None]
            reverse = k in higher_is_better
            valid.sort(key=lambda x: (-x[1] if reverse else x[1]))
            # Assign ranks with ties -> average rank
            ranked = {}
            i = 0
            while i < len(valid):
                j = i
                while j + 1 < len(valid) and valid[j + 1][1] == valid[i][1]:
                    j += 1
                avg_rank = (i + 1 + j + 1) / 2.0
                for kk in range(i, j + 1):
                    ranked[valid[kk][0]] = avg_rank
                i = j + 1
            for m in invalid:
                ranked[m] = None
            rankings[s][k] = ranked
    return rankings


def pareto_frontier(stratified_per_method: dict, strata_keys: list,
                    q_metric: str, plaus_metric: str,
                    higher_is_better: set) -> dict:
    """For each stratum, return methods on the Pareto frontier.

    We minimise q_metric (unless in higher_is_better) and minimise plaus_metric
    (both skating and accel_rms are lower-is-better).
    """
    out = {}
    methods = list(stratified_per_method.keys())
    for s in strata_keys:
        pts = []
        for m in methods:
            stats = stratified_per_method[m].get(s, {})
            q_v = stats.get(q_metric)
            p_v = stats.get(plaus_metric)
            if q_v is None or p_v is None:
                continue
            if q_metric in higher_is_better:
                q_v = -q_v  # convert to minimisation
            pts.append((m, q_v, p_v))
        frontier = []
        for i, (m, q_i, p_i) in enumerate(pts):
            dominated = False
            for j, (m2, q_j, p_j) in enumerate(pts):
                if i == j:
                    continue
                if q_j <= q_i and p_j <= p_i and (q_j < q_i or p_j < p_i):
                    dominated = True
                    break
            if not dominated:
                frontier.append({'method': m,
                                 'q_metric_value': q_i if q_metric not in higher_is_better else -q_i,
                                 'plausibility_value': p_i})
        out[s] = frontier
    return out


def run():
    t0 = time.time()
    from eval.quotient_extractor import extract_quotient  # noqa

    print('Loading caches...')
    with open(EVAL_PAIRS) as f:
        eval_spec = json.load(f)
    pairs_by_id = {int(p['pair_id']): p for p in eval_spec['pairs']}

    cond_dict = np.load(COND_PATH, allow_pickle=True).item()
    with open(CONTACT_GROUPS_PATH) as f:
        contact_groups = json.load(f)

    methods = discover_methods()
    print(f'Discovered {len(methods)} methods:')
    for m in methods:
        print(f'  - {m}')

    # Extract source Q once per source fname (cache)
    source_q_cache = {}
    per_method_per_pair = {}
    for method in methods:
        method_dir = K_COMPARE_DIR / method
        pair_files = discover_method_pairs(method_dir)
        print(f'\n[{method}] {len(pair_files)} pair files')
        per_method_per_pair[method] = []

        for pid, fname in pair_files:
            if pid not in pairs_by_id:
                print(f'  skip {fname}: pair_id {pid} not in eval_spec')
                continue
            meta = pairs_by_id[pid]
            src_fname = meta['source_fname']
            src_skel = meta['source_skel']
            tgt_skel = meta['target_skel']
            support = int(meta['support_same_label'])
            family_gap = meta['family_gap']
            strata = stratum_of(family_gap, support)

            entry = {
                'pair_id': pid, 'src_fname': src_fname, 'src_skel': src_skel,
                'tgt_skel': tgt_skel, 'family_gap': family_gap,
                'support_same_label': support, 'strata': strata,
                'status': 'pending', 'error': None,
            }

            try:
                if src_fname not in source_q_cache:
                    source_q_cache[src_fname] = compute_source_q(
                        src_fname, cond_dict, contact_groups, extract_quotient)
                q_src = source_q_cache[src_fname]

                if tgt_skel not in cond_dict:
                    raise RuntimeError(f'missing cond for tgt skel {tgt_skel}')

                npy_path = method_dir / fname
                motion = np.load(npy_path)
                q_tgt = compute_target_q_from_npy(
                    npy_path, tgt_skel, cond_dict, contact_groups, extract_quotient)

                q_d = q_component_distances(q_src, q_tgt)
                plaus = compute_plausibility(motion, cond_dict[tgt_skel], fps=FPS)

                entry.update(q_d)
                entry.update(plaus)
                entry['status'] = 'ok'
            except Exception as e:
                entry['status'] = 'failed'
                entry['error'] = f'{e}\n{traceback.format_exc(limit=2)}'
            per_method_per_pair[method].append(entry)
        ok = sum(1 for e in per_method_per_pair[method] if e['status'] == 'ok')
        print(f'  done: {ok}/{len(pair_files)} ok')

    metric_keys = [
        'q_com_path_l2', 'q_heading_vel_l2', 'q_contact_sched_aggregate_l2',
        'q_cadence_abs_diff', 'q_limb_usage_top5_l2',
        'contact_f1_vs_source_aggregate',
        'skating_proxy', 'accel_rms',
    ]
    higher_is_better = {'contact_f1_vs_source_aggregate'}
    strata_keys = ['near_present', 'absent', 'moderate', 'extreme', 'all']

    print('\nComputing stratified means...')
    stratified_per_method = {}
    for method in methods:
        ok_entries = [e for e in per_method_per_pair[method] if e['status'] == 'ok']
        stratified_per_method[method] = stratified_means(
            ok_entries, strata_keys, metric_keys)

    print('Computing rankings...')
    rankings = compute_rankings(stratified_per_method, strata_keys,
                                metric_keys, higher_is_better)

    print('Computing Pareto frontier (mean-Q-L2 vs accel_rms)...')
    # Add a "mean Q L2" derived metric for Pareto and also for rank headline
    q_l2_keys = [
        'q_com_path_l2', 'q_heading_vel_l2', 'q_contact_sched_aggregate_l2',
        'q_limb_usage_top5_l2',
    ]
    for method in methods:
        for s in strata_keys:
            bucket = stratified_per_method[method].get(s, {})
            vals = [bucket[k] for k in q_l2_keys if bucket.get(k) is not None]
            bucket['mean_q_l2'] = float(np.mean(vals)) if vals else None
            stratified_per_method[method][s] = bucket

    pareto_accel = pareto_frontier(stratified_per_method, strata_keys,
                                   'mean_q_l2', 'accel_rms', higher_is_better)
    pareto_skat = pareto_frontier(stratified_per_method, strata_keys,
                                  'mean_q_l2', 'skating_proxy', higher_is_better)
    pareto_f1_accel = pareto_frontier(stratified_per_method, strata_keys,
                                      'contact_f1_vs_source_aggregate',
                                      'accel_rms', higher_is_better)

    # Extend rankings with mean_q_l2
    rankings_extra = compute_rankings(stratified_per_method, strata_keys,
                                      ['mean_q_l2'], higher_is_better)
    for s in strata_keys:
        rankings[s]['mean_q_l2'] = rankings_extra[s]['mean_q_l2']

    out = {
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'eval_pairs': str(EVAL_PAIRS),
        'n_methods': len(methods),
        'methods': methods,
        'metric_keys': metric_keys + ['mean_q_l2'],
        'higher_is_better': sorted(list(higher_is_better)),
        'strata_keys': strata_keys,
        'per_method_per_pair': per_method_per_pair,
        'stratified_means': stratified_per_method,
        'rankings': rankings,
        'pareto': {
            'mean_q_l2__vs__accel_rms': pareto_accel,
            'mean_q_l2__vs__skating_proxy': pareto_skat,
            'contact_f1_vs_source_aggregate__vs__accel_rms': pareto_f1_accel,
        },
        'runtime_s': float(time.time() - t0),
    }
    with open(OUT_PATH, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f'\nSaved: {OUT_PATH} ({(time.time()-t0):.1f}s total)')

    # Print headline table: method x stratum on contact_f1_vs_source_aggregate
    print('\n=== Headline: contact_f1_vs_source_aggregate by method x stratum ===')
    hdr = 'method,' + ','.join(strata_keys)
    print(hdr)
    for m in methods:
        cells = [m]
        for s in strata_keys:
            v = stratified_per_method[m].get(s, {}).get('contact_f1_vs_source_aggregate')
            cells.append(f'{v:.3f}' if v is not None else '-')
        print(','.join(cells))

    print('\n=== Headline: mean_q_l2 by method x stratum (lower=better) ===')
    print(hdr)
    for m in methods:
        cells = [m]
        for s in strata_keys:
            v = stratified_per_method[m].get(s, {}).get('mean_q_l2')
            cells.append(f'{v:.3f}' if v is not None else '-')
        print(','.join(cells))

    return out


if __name__ == '__main__':
    run()
