"""M2M Official baseline on v3 benchmark.

Wraps `external/motion2motion/run_M2M.py` (the official Motion2Motion code,
arXiv:2508.13139) for v3 manifests.

For each query:
  1. Use source motion BVH and a chosen example (target_skel) BVH
  2. Auto-generate mapping JSON from contact_groups.json
  3. Run official M2M → output BVH
  4. Convert output BVH → joint positions [T, J, 3] (via FK)
  5. Save as [T, J, 3] (NOT 13-dim — positions only)

Per Codex review:
  - Scaffold pool excludes positives + adversarials (no leakage)
  - Per-query deterministic seed (not iteration-order-dependent)
  - Output length matched to pos_median_T

Output: eval/results/baselines/m2m_official_v5/fold_{42,43}/query_NNNN.npy [T, J, 3]
"""
from __future__ import annotations
ANYTOP_DATA = Path(os.environ.get("ANYTOP_DATA", "")) or PROJECT_ROOT / "dataset"
import argparse
import hashlib
import json
import os
import sys
import shutil
import subprocess
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BVH_DIR = PROJECT_ROOT / 'dataset/truebones/zoo/truebones_processed/bvhs'
M2M_DIR = PROJECT_ROOT / 'external/motion2motion'


def list_target_skel_bvhs(skel_name):
    """All .bvh files for given target skeleton."""
    return sorted([f for f in os.listdir(BVH_DIR)
                   if (f.startswith(skel_name + '___') or f.startswith(skel_name + '_'))
                   and f.endswith('.bvh')])


def load_bvh_positions(bvh_path):
    """Load BVH and return joint positions [T, J, 3] via forward kinematics."""
    import BVH
    from data_loaders.truebones.truebones_utils.motion_process import positions_global
    anim, names, _ = BVH.load(str(bvh_path))
    pos = positions_global(anim)  # [T, J, 3]
    return pos


def get_m2m_visible_names(bvh_path):
    """Parse BVH to get the joint names M2M will see (ROOT/JOINT, skipping End Site).
    Returns list of joint names in the order M2M's parser sees them."""
    import re
    visible = []
    skip_until_brace = 0  # depth counter for End Site blocks
    in_end_site = False
    with open(bvh_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith('End Site'):
                in_end_site = True
                continue
            if in_end_site:
                # Wait for closing brace of End Site
                if stripped == '}':
                    in_end_site = False
                continue
            m = re.match(r'^(JOINT|ROOT)\s+([A-Za-z0-9_\-:]+)', stripped)
            if m:
                visible.append(m.group(2))
            if stripped.startswith('MOTION'):
                break  # done with hierarchy
    return visible


# Per-skeleton normalizer cache: skel → (cond_to_visible: dict, visible_names: list)
_NORMALIZER_CACHE = {}


def get_normalizer(skel_name, cond):
    """Return (cond_to_visible, visible_names) for a skeleton.
    cond_to_visible: dict mapping cond_joint_idx → nearest ancestor name that M2M can see.
    """
    if skel_name in _NORMALIZER_CACHE:
        return _NORMALIZER_CACHE[skel_name]
    # Find a representative BVH for this skeleton
    bvhs = sorted([f for f in os.listdir(BVH_DIR)
                   if (f.startswith(skel_name + '___') or f.startswith(skel_name + '_'))
                   and f.endswith('.bvh')])
    if not bvhs:
        raise RuntimeError(f'No BVH for {skel_name}')
    rep_bvh = BVH_DIR / bvhs[0]
    visible = get_m2m_visible_names(rep_bvh)
    visible_set = set(visible)
    parents = cond[skel_name]['parents']
    names = list(cond[skel_name]['joints_names'])
    cond_to_visible = {}
    for i, name in enumerate(names):
        # Walk up parents until we find a visible name
        cur = i
        steps = 0
        while steps < 50:  # safety
            cur_name = names[cur]
            if cur_name in visible_set:
                cond_to_visible[i] = cur_name
                break
            p = parents[cur]
            if p < 0 or p == cur:
                # Reached root with no match — fall back to first visible (root)
                cond_to_visible[i] = visible[0] if visible else cur_name
                break
            cur = p
            steps += 1
    _NORMALIZER_CACHE[skel_name] = (cond_to_visible, visible)
    return cond_to_visible, visible


def preprocess_bvh_with_suffix(src_bvh, dst_bvh):
    """Add `__XXX` random suffix to each joint name in BVH, save to dst_bvh.
    M2M's parser strips last 5 chars when comparing to mapping JSON names."""
    import re, random, string
    with open(src_bvh, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    rng = random.Random(42)  # deterministic suffix per file
    joint_pat = re.compile(r"^(\s*)(JOINT|ROOT)\s+([A-Za-z0-9_\-:]+)(\s*)$")
    out_lines = []
    for line in lines:
        m = joint_pat.match(line)
        if m:
            spaces, jtype, jname, trail = m.groups()
            suffix = ''.join(rng.choices(string.ascii_letters + string.digits, k=3))
            out_lines.append(f"{spaces}{jtype} {jname}__{suffix}{trail}")
        else:
            out_lines.append(line)
    with open(dst_bvh, 'w', encoding='utf-8') as f:
        f.writelines(out_lines)


def auto_generate_mapping(skel_a, skel_b, cond, contact_groups, max_pairs=8):
    """Auto-generate M2M mapping JSON.

    Uses contact_groups heuristic, then NORMALIZES joint names via per-skel
    cond→M2M-visible map (parent walk for joints M2M can't see).
    """
    from eval.motion2motion_run import author_sparse_mapping

    pairs_ij, _ = author_sparse_mapping(skel_a, skel_b, cond, contact_groups,
                                        max_pairs=max_pairs)

    src_to_vis, src_visible = get_normalizer(skel_a, cond)
    tgt_to_vis, tgt_visible = get_normalizer(skel_b, cond)

    # Build root→root pair (always include — M2M assumes index 0 is root)
    root_pair = (src_visible[0], tgt_visible[0])

    # Build mapping: normalize each pair, dedupe
    mapping_set = set()
    mapping_set.add(root_pair)
    for src_idx, tgt_idx in pairs_ij:
        s_name = src_to_vis.get(src_idx)
        t_name = tgt_to_vis.get(tgt_idx)
        if s_name is None or t_name is None:
            continue
        mapping_set.add((s_name, t_name))

    mapping = [{'source': s, 'target': t} for (s, t) in sorted(mapping_set)]

    return {
        'source_name': skel_a,
        'target_name': skel_b,
        'root_joint': src_visible[0],
        'mapping': mapping,
    }


def run_m2m_official(src_bvh, ex_bvh, mapping_json, output_dir, device='cpu',
                     timeout_sec=300):
    """Call official M2M run_M2M.py on a single (source, example) pair.

    Returns: path to output BVH file, or None on failure.
    """
    cmd = [
        f"{ANYTOP_DATA}/miniconda3/envs/anytop/bin/python",
        '-u', 'run_M2M.py',
        '-e', str(ex_bvh),  # example/target BVH
        '-d', device,
        '--source', str(src_bvh),
        '--mapping_file', str(mapping_json),
        '--output_dir', str(output_dir),
        '--sparse_retargeting',
        '--matching_alpha', '0.9',
    ]
    try:
        result = subprocess.run(cmd, cwd=str(M2M_DIR),
                                capture_output=True, text=True, timeout=timeout_sec)
        if result.returncode != 0:
            err_summary = (result.stderr[-800:] if result.stderr else '') + ' || STDOUT: ' + (result.stdout[-400:] if result.stdout else '')
            return None, err_summary
        # M2M outputs to <output_dir>/<example_basename>/<prefix>_syn.bvh — walk recursively
        out_bvhs = []
        for root_dir, _, files in os.walk(output_dir):
            for f in files:
                if f.endswith('_syn.bvh'):
                    out_bvhs.append(Path(root_dir) / f)
        if not out_bvhs:
            return None, 'No output BVH (rc=0, stdout: ' + (result.stdout[-400:] if result.stdout else 'empty') + ')'
        # Most recent
        out_bvhs.sort(key=lambda p: p.stat().st_mtime)
        return out_bvhs[-1], None
    except subprocess.TimeoutExpired:
        return None, f'Timeout after {timeout_sec}s'


def per_query_seed(fold, qid, skel_b):
    s = f"{fold}_{qid}_{skel_b}".encode()
    return int(hashlib.md5(s).hexdigest()[:8], 16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=42)
    parser.add_argument('--manifest', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--max_queries', type=int, default=10000)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--timeout', type=int, default=300)
    args = parser.parse_args()

    if args.manifest:
        manifest_path = Path(args.manifest)
    else:
        manifest_path = PROJECT_ROOT / f'eval/benchmark_v3/queries_v5/fold_{args.fold}/manifest.json'
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = PROJECT_ROOT / f'eval/results/baselines/m2m_official_v5/fold_{args.fold}'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Manifest: {manifest_path}")
    print(f"Output dir: {out_dir}")
    print(f"Device: {args.device}, timeout: {args.timeout}s")

    from eval.run_k_pipeline_200pairs import load_assets
    cond, contact_groups, _ = load_assets()

    with open(manifest_path) as f:
        manifest = json.load(f)
    queries = manifest['queries'][:args.max_queries]
    print(f"Running {len(queries)} queries")

    # Per-query temp dirs cleaned up
    work_root = Path(tempfile.mkdtemp(prefix='m2m_official_'))
    print(f"Working dir: {work_root}")

    tgt_bvh_cache = {}
    per_query = []
    t_total = time.time()

    for i, q in enumerate(queries):
        qid = q['query_id']
        skel_a = q['skel_a']
        skel_b = q['skel_b']
        src_fname = q['src_fname']
        cluster = q['cluster']
        split = q['split']

        rec = {'query_id': qid, 'cluster': cluster, 'split': split,
               'skel_a': skel_a, 'skel_b': skel_b, 'status': 'pending'}

        try:
            # 1. Source BVH
            src_bvh_name = src_fname.replace('.npy', '.bvh')
            src_bvh = BVH_DIR / src_bvh_name
            if not src_bvh.exists():
                rec['status'] = 'skipped_no_src_bvh'
                per_query.append(rec)
                continue

            # 2. Pick example BVH (scaffold) — exclude positives + adversarials
            if skel_b not in tgt_bvh_cache:
                tgt_bvh_cache[skel_b] = list_target_skel_bvhs(skel_b)
            full_pool = tgt_bvh_cache[skel_b]
            # V5 schema: union of all candidate clip families used by the evaluator
            forbidden = set()
            for key in ('positives_cluster', 'positives_exact',
                        'adversarials_easy', 'adversarials_hard',
                        'distractors_same_target_skel'):
                for x in q.get(key, []):
                    forbidden.add(x['fname'].replace('.npy', '.bvh'))
            pool = [f for f in full_pool if f not in forbidden]
            if not pool:
                pool = full_pool
                rec['scaffold_fallback'] = 'positive_excluded_empty'
            qseed = per_query_seed(args.fold, qid, skel_b)
            qrng = np.random.RandomState(qseed)
            ex_bvh_name = pool[qrng.randint(0, len(pool))]
            ex_bvh = BVH_DIR / ex_bvh_name

            # 3. Generate mapping JSON
            if skel_a not in contact_groups or skel_b not in contact_groups:
                rec['status'] = 'skipped_no_cg'
                per_query.append(rec)
                continue
            mapping = auto_generate_mapping(skel_a, skel_b, cond, contact_groups)
            # Per Codex: require ≥3 unique pairs (root + 2 non-root) for stable retargeting
            if not mapping['mapping'] or len(mapping['mapping']) < 3:
                rec['status'] = 'skipped_few_pairs'
                rec['n_pairs'] = len(mapping['mapping']) if mapping else 0
                per_query.append(rec)
                continue

            # Per-query work dir
            q_work = work_root / f'q_{qid:04d}'
            q_work.mkdir(exist_ok=True)
            # Preprocess BVHs with __XXX suffixes (M2M parser strips last 5 chars)
            src_bvh_pp = q_work / src_bvh.name
            ex_bvh_pp = q_work / ex_bvh.name
            preprocess_bvh_with_suffix(src_bvh, src_bvh_pp)
            preprocess_bvh_with_suffix(ex_bvh, ex_bvh_pp)
            mapping_json = q_work / 'mapping.json'
            with open(mapping_json, 'w') as f:
                json.dump(mapping, f, indent=2)
            m2m_out_dir = q_work / 'output'
            m2m_out_dir.mkdir(exist_ok=True)

            # 4. Run M2M
            t0 = time.time()
            output_bvh, err = run_m2m_official(
                src_bvh_pp, ex_bvh_pp, mapping_json, m2m_out_dir,
                device=args.device, timeout_sec=args.timeout)
            runtime = time.time() - t0
            if output_bvh is None:
                rec['status'] = 'm2m_failed'
                rec['error'] = err
                per_query.append(rec)
                continue

            # 5. Load output BVH and extract positions
            positions = load_bvh_positions(output_bvh)  # [T_out, J_out, 3]

            # 6. V5: median of positives_cluster T (no pos_median_T field)
            pos_T = [p['T'] for p in q.get('positives_cluster', [])]
            if not pos_T:
                target_len = q.get('src_T', positions.shape[0])
            else:
                target_len = int(np.median(pos_T))
            if positions.shape[0] != target_len:
                from scipy.interpolate import interp1d
                T_in = positions.shape[0]
                xs_in = np.linspace(0, 1, T_in)
                xs_out = np.linspace(0, 1, target_len)
                positions_resamp = np.zeros((target_len,) + positions.shape[1:],
                                             dtype=np.float32)
                for j in range(positions.shape[1]):
                    for c in range(positions.shape[2]):
                        f_interp = interp1d(xs_in, positions[:, j, c], kind='linear',
                                            assume_sorted=True)
                        positions_resamp[:, j, c] = f_interp(xs_out)
                positions = positions_resamp

            # 7. Save as [T, J, 3] — eval will detect shape and skip _recover_positions
            np.save(out_dir / f'query_{qid:04d}.npy', positions.astype(np.float32))
            rec['status'] = 'ok'
            rec['scaffold_bvh'] = ex_bvh_name
            rec['n_pairs'] = len(mapping['mapping'])
            rec['runtime_sec'] = runtime
            rec['T_out'] = int(positions.shape[0])
            rec['J_out'] = int(positions.shape[1])

            # Cleanup per-query work dir to save disk
            shutil.rmtree(q_work, ignore_errors=True)

            if (i + 1) % 5 == 0 or i == 0:
                elapsed = time.time() - t_total
                eta = elapsed / (i + 1) * (len(queries) - i - 1)
                print(f"  [{i+1}/{len(queries)}] {cluster}/{split} {skel_a}→{skel_b} "
                      f"T={rec['T_out']} J={rec['J_out']} ({runtime:.1f}s, ETA {eta:.0f}s)")

        except Exception as e:
            rec['status'] = 'failed'
            rec['error'] = str(e) + '\n' + traceback.format_exc(limit=3)
            print(f"  FAILED query {qid}: {e}")

        per_query.append(rec)

    # Cleanup
    shutil.rmtree(work_root, ignore_errors=True)

    total_time = time.time() - t_total
    n_ok = sum(1 for r in per_query if r['status'] == 'ok')
    n_failed = sum(1 for r in per_query if r['status'] in ('failed', 'm2m_failed'))
    n_skipped = sum(1 for r in per_query if 'skipped' in r['status'])
    print(f"\nTotal: {total_time:.0f}s, OK: {n_ok}/{len(per_query)}, "
          f"failed: {n_failed}, skipped: {n_skipped}")

    summary = {
        'method': 'M2M_official_v3',
        'manifest': str(manifest_path),
        'n_queries': len(per_query),
        'n_ok': n_ok,
        'n_failed': n_failed,
        'n_skipped': n_skipped,
        'total_time_sec': total_time,
        'output_format': 'positions_T_J_3',  # NOT 13-dim
        'per_query': per_query,
    }
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved: {out_dir}/metrics.json")


if __name__ == '__main__':
    main()
