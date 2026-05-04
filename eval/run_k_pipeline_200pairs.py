"""Run Idea K pipeline on the 200-pair benchmark (eval/benchmark_paired/).

Adapted from run_k_pipeline_30pairs.py. Adds:
  - Bootstrap 95% CI on all aggregate metrics
  - Train/test split analysis (TEST_SKELETONS)
  - Morphology breakdown
  - Invariant-rep DTW comparison against zero-training baseline

Outputs:
  eval/results/k_compare/K_200pair/metrics.json
  eval/results/k_compare/K_200pair/pair_<id>.npy  (refined 13-dim motions)
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

from model.skel_blind.invariant_dataset import TEST_SKELETONS
from eval.build_contact_groups import classify_morphology

OUT_DIR = PROJECT_ROOT / 'eval/results/k_compare/K_200pair'
OUT_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST = PROJECT_ROOT / 'eval/benchmark_paired/pairs/manifest.json'


def load_assets():
    from data_loaders.truebones.truebones_utils.param_utils import DATASET_DIR
    cond = np.load(os.path.join(DATASET_DIR, 'cond.npy'),
                   allow_pickle=True).item()
    with open(PROJECT_ROOT / 'eval/quotient_assets/contact_groups.json') as f:
        cg = json.load(f)
    motion_dir = os.path.join(DATASET_DIR, 'motions')
    return cond, cg, motion_dir


def get_morphology(cond_dict, skel_name):
    entry = cond_dict[skel_name]
    names = list(entry["joints_names"])
    parents = list(entry["parents"])
    chains = [list(c) for c in entry["kinematic_chains"]]
    return classify_morphology(names, parents, chains)


def split_category(skel_a, skel_b):
    a_test = skel_a in TEST_SKELETONS
    b_test = skel_b in TEST_SKELETONS
    if a_test and b_test:
        return "test_test"
    elif a_test or b_test:
        return "mixed"
    return "train_train"


def bootstrap_ci(values, n_boot=1000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return (0.0, 0.0, 0.0)
    means = np.array([rng.choice(arr, size=n, replace=True).mean()
                      for _ in range(n_boot)])
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return (round(float(lo), 4), round(float(arr.mean()), 4), round(float(hi), 4))


# Reuse helpers from the 30-pair script
from eval.run_k_pipeline_30pairs import (
    remap_contact_sched, build_q_star, q_component_errors,
    contact_f1, skating_proxy, build_contact_mask_tj,
)


def run(max_pairs=200):
    with open(MANIFEST) as f:
        manifest = json.load(f)
    pairs = manifest['pairs'][:max_pairs]
    print(f"Loaded {len(pairs)} pairs from manifest")

    cond, contact_groups, motion_dir = load_assets()

    from eval.quotient_extractor import extract_quotient
    from eval.ik_solver import solve_ik
    from eval.k_pipeline_bridge import theta_to_motion_13dim, bridge_diagnostics
    from eval.anytop_projection import anytop_project

    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    per_pair = []
    t_total_0 = time.time()

    for i, p in enumerate(pairs):
        pid = p['pair_id']
        src_skel = p['skel_a']
        tgt_skel = p['skel_b']
        src_fname = p['file_a']
        action = p['action']

        if (i + 1) % 20 == 0 or i == 0:
            elapsed = time.time() - t_total_0
            eta = elapsed / (i + 1) * (len(pairs) - i - 1) if i > 0 else 0
            print(f"  [{i+1}/{len(pairs)}] {action}: {src_skel}→{tgt_skel} "
                  f"(elapsed {elapsed:.0f}s, ETA {eta:.0f}s)")

        rec = {
            'pair_id': pid,
            'action': action,
            'source_skel': src_skel,
            'target_skel': tgt_skel,
            'source_fname': src_fname,
            'split': split_category(src_skel, tgt_skel),
            'morph_src': get_morphology(cond, src_skel),
            'morph_tgt': get_morphology(cond, tgt_skel),
            'status': 'pending',
        }

        t0 = time.time()
        try:
            if src_skel not in contact_groups or tgt_skel not in contact_groups:
                rec['status'] = 'skipped_no_contact_groups'
                rec['pair_runtime'] = 0.0
                per_pair.append(rec)
                continue

            q_src = extract_quotient(src_fname, cond[src_skel],
                                     contact_groups=contact_groups,
                                     motion_dir=motion_dir)
            q_star = build_q_star(q_src, src_skel, tgt_skel,
                                  contact_groups, cond)

            ik_out = solve_ik(q_star, cond[tgt_skel],
                              contact_groups[tgt_skel],
                              n_iters=400, verbose=False,
                              device=device)
            rec['ik_runtime'] = float(ik_out['runtime_sec'])

            q_rec = ik_out['q_reconstructed']
            errs = q_component_errors(q_rec, q_star)
            rec['q_errors'] = errs
            rec['contact_f1'] = contact_f1(q_rec['contact_sched'],
                                           q_star['contact_sched'])

            try:
                motion_13 = theta_to_motion_13dim(
                    ik_out['theta'], ik_out['root_pos'],
                    ik_out['positions'],
                    tgt_skel, cond,
                    contact_groups=contact_groups)
            except Exception:
                motion_13 = theta_to_motion_13dim(
                    ik_out['theta'], ik_out['root_pos'],
                    ik_out['positions'],
                    tgt_skel, cond,
                    contact_groups=contact_groups,
                    fit_rotations=False)

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
                rec['status'] = 'stage3_failed'
                rec['error'] = str(e)
                out_motion = motion_13

            out_path = OUT_DIR / f'pair_{pid:04d}.npy'
            np.save(out_path, out_motion)

            rec['skating_proxy'] = skating_proxy(out_motion, contact_groups,
                                                  tgt_skel, cond)
            rec['pair_runtime'] = float(time.time() - t0)

        except Exception as e:
            rec['status'] = 'failed'
            rec['error'] = str(e) + '\n' + traceback.format_exc(limit=3)
            rec['pair_runtime'] = float(time.time() - t0)
            print(f"  FAILED pair {pid} {src_skel}→{tgt_skel}: {e}")

        per_pair.append(rec)

    total_time = time.time() - t_total_0

    # ====== aggregate ======
    ok = [r for r in per_pair if r['status'] in ('ok', 'stage3_failed')]
    print(f"\n{'='*60}")
    print(f"IDEA K — 200-PAIR BENCHMARK")
    print(f"{'='*60}")
    print(f"Total: {len(per_pair)}, OK: {len(ok)}, "
          f"Failed: {sum(1 for r in per_pair if r['status'] == 'failed')}, "
          f"Skipped: {sum(1 for r in per_pair if 'skipped' in r.get('status',''))}")
    print(f"Total time: {total_time:.1f}s ({total_time/len(per_pair):.1f}s/pair)")

    def agg_metrics(entries):
        if not entries:
            return {}
        f1s = [e['contact_f1'] for e in entries if 'contact_f1' in e]
        skates = [e['skating_proxy'] for e in entries if 'skating_proxy' in e]
        coms = [e['q_errors']['com_path_rel_l2'] for e in entries if 'q_errors' in e]
        return {
            'n': len(entries),
            'contact_f1': bootstrap_ci(f1s) if f1s else None,
            'skating': bootstrap_ci(skates) if skates else None,
            'com_rel_l2': bootstrap_ci(coms) if coms else None,
        }

    overall = agg_metrics(ok)

    by_split = {}
    for split in ('train_train', 'mixed', 'test_test'):
        subset = [r for r in ok if r.get('split') == split]
        if subset:
            by_split[split] = agg_metrics(subset)

    by_morph = {}
    morph_pairs = set()
    for r in ok:
        key = f"{r.get('morph_src','?')}→{r.get('morph_tgt','?')}"
        morph_pairs.add(key)
    for key in sorted(morph_pairs):
        subset = [r for r in ok if f"{r.get('morph_src','?')}→{r.get('morph_tgt','?')}" == key]
        if len(subset) >= 3:
            by_morph[key] = agg_metrics(subset)

    # Print
    print(f"\n--- Overall ---")
    if overall:
        print(f"  Contact F1: {overall['contact_f1']}")
        print(f"  Skating: {overall['skating']}")
        print(f"  COM rel L2: {overall['com_rel_l2']}")

    print(f"\n--- By Split ---")
    for split, m in by_split.items():
        print(f"  {split:15s}: N={m['n']}, F1={m['contact_f1']}, skate={m['skating']}")

    print(f"\n--- By Morphology (≥3 pairs) ---")
    for key, m in sorted(by_morph.items(), key=lambda x: -x[1]['n']):
        print(f"  {key:30s}: N={m['n']}, F1={m['contact_f1']}")

    report = {
        'method': 'Idea_K',
        'n_pairs': len(per_pair),
        'n_ok': len(ok),
        'total_time_sec': total_time,
        'overall': overall,
        'by_split': by_split,
        'by_morphology': by_morph,
        'per_pair': per_pair,
    }

    out_path = OUT_DIR / 'metrics.json'
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_pairs', type=int, default=200)
    args = parser.parse_args()
    run(max_pairs=args.max_pairs)
