"""Full unified comparison with the v2 external classifier.

Extends `eval/k_action_accuracy_v2.py`:
  - Auto-discovers every method directory under eval/results/k_compare/ that
    contains at least one `pair_*.npy` file (so idea_F_topo_lora, K_multi_init,
    idea_O_dtw_morph etc. are picked up automatically when they appear).
  - Reports per-class prediction counts for every method (exposes any
    "classifier predicts everything as idle/other" pathology).
  - Writes to idea-stage/unified_method_comparison_v2_full.json (distinct
    filename from the earlier abbreviated comparison).

Protocol is identical to `k_action_accuracy.py`: for each pair, run the
classifier on the retargeted motion and compare the predicted label both to
(a) the source label (label_match) and (b) the classifier's prediction on
the source motion (behavior_preserved).
"""
from __future__ import annotations

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(str(PROJECT_ROOT))
sys.path.insert(0, str(ROOT))

EVAL_PAIRS = ROOT / 'idea-stage/eval_pairs.json'
COMPARE_DIR = ROOT / 'eval/results/k_compare'
CLF_CKPT = ROOT / 'save/external_classifier_v2.pt'
OUT = ROOT / 'idea-stage/unified_method_comparison_v2_full.json'

DATASET_DIR = ROOT / 'dataset/truebones/zoo/truebones_processed'

# Canonical order for reporting (kept stable across runs). Any directory under
# COMPARE_DIR that holds pair_*.npy files but isn't in this list will still be
# evaluated; it just gets appended at the end.
PREFERRED_ORDER = [
    'K', 'K_no_stage3', 'K_no_stage2',
    'retrieve_refine', 'retrieve_refine_v2', 'retrieve_refine_v3', 'retrieve_refine_v4',
    'label_random', 'psi_retrieval', 'q_retrieval', 'q_label_retrieval',
    'motion2motion', 'A_full', 'A_no_action',
    # Newer methods the plan calls for (only evaluated if their dirs exist):
    'idea_F_topo_lora', 'K_multi_init', 'K_ikprior', 'idea_O_dtw_morph',
]


def discover_methods():
    """Return list of method directory names that actually contain pair_*.npy."""
    if not COMPARE_DIR.exists():
        return []
    found = []
    for entry in sorted(COMPARE_DIR.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name == 'renders':
            continue
        has_pair = any(f.name.startswith('pair_') and f.name.endswith('.npy')
                       for f in entry.iterdir())
        if has_pair:
            found.append(entry.name)
    # Order: preferred first (only those that exist), then any extras alphabetical.
    ordered = [m for m in PREFERRED_ORDER if m in found]
    extras = sorted([m for m in found if m not in PREFERRED_ORDER])
    return ordered + extras


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    from eval.external_classifier import extract_classifier_features, ACTION_CLASSES
    from eval.train_external_classifier_v2 import V2Classifier
    from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np

    clf = V2Classifier(str(CLF_CKPT), device=device)
    print(
        f"Loaded v2 classifier (arch={clf.arch}, target_T={clf.target_T}, "
        f"feat_mean shape={tuple(clf.feat_mean.shape)})",
        flush=True,
    )

    cond = np.load(DATASET_DIR / 'cond.npy', allow_pickle=True).item()

    with open(EVAL_PAIRS) as f:
        eval_set = json.load(f)
    pairs = eval_set['pairs']

    def classify_motion(motion_13dim, skel):
        J_skel = cond[skel]['offsets'].shape[0]
        m = motion_13dim
        if m.shape[1] > J_skel:
            m = m[:, :J_skel]
        if np.abs(m).max() < 5:
            mean = cond[skel]['mean'][:J_skel]
            std = cond[skel]['std'][:J_skel]
            m = m.astype(np.float32) * std + mean
        try:
            positions = recover_from_bvh_ric_np(m.astype(np.float32))
        except Exception as e:
            return None, str(e)
        parents = cond[skel]['parents'][:J_skel]
        feats = extract_classifier_features(positions, parents)
        if feats is None or feats.shape[0] < 4:
            return None, 'feats_none'
        pred = clf.predict_label(feats)
        return pred, None

    # Source classifier predictions (per source fname)
    print("\nComputing source classifier predictions...", flush=True)
    src_pred_by_pair = {}
    src_stride_cache = {}
    for p in pairs:
        src_fname = p['source_fname']
        if src_fname in src_stride_cache:
            src_pred_by_pair[p['pair_id']] = src_stride_cache[src_fname]
            continue
        m = np.load(DATASET_DIR / 'motions' / src_fname)
        pred, err = classify_motion(m, p['source_skel'])
        src_stride_cache[src_fname] = pred
        src_pred_by_pair[p['pair_id']] = pred
    # Flag any source clip that didn't classify (rare but possible when
    # feature extraction fails). We keep the pair but behavior_preserved stays
    # False for it.
    n_src_miss = sum(1 for v in src_pred_by_pair.values() if v is None)
    if n_src_miss:
        print(f"  WARNING: {n_src_miss} source clips failed to classify.", flush=True)

    methods = discover_methods()
    print(f"\nDiscovered {len(methods)} method dirs: {methods}", flush=True)

    all_results = {}
    for method in methods:
        method_dir = COMPARE_DIR / method
        per_pair = []
        for p in pairs:
            pid = p['pair_id']
            patterns = [f for f in os.listdir(method_dir)
                        if f.startswith(f'pair_{pid:02d}_') and f.endswith('.npy')]
            if not patterns:
                per_pair.append({'pair_id': pid, 'error': 'no_output'})
                continue
            motion_path = method_dir / patterns[0]
            try:
                motion = np.load(motion_path)
            except Exception as e:
                per_pair.append({'pair_id': pid, 'error': f'load: {e}'})
                continue
            pred, err = classify_motion(motion, p['target_skel'])
            if pred is None:
                per_pair.append({'pair_id': pid, 'error': err or 'classify_none'})
                continue
            src_pred = src_pred_by_pair.get(pid)
            per_pair.append({
                'pair_id': pid,
                'src_skel': p['source_skel'],
                'tgt_skel': p['target_skel'],
                'src_label': p['source_label'],
                'family_gap': p['family_gap'],
                'support_same_label': p['support_same_label'],
                'tgt_pred': pred,
                'src_classifier_pred': src_pred,
                'label_match': pred == p['source_label'],
                'behavior_preserved': src_pred is not None and pred == src_pred,
            })
        valid = [r for r in per_pair if 'error' not in r]
        print(f"  {method}: valid {len(valid)}/{len(per_pair)}", flush=True)

        # Stratify: absent takes priority (any pair with support_same_label==0),
        # otherwise family_gap.
        strata = defaultdict(list)
        for r in valid:
            if r['support_same_label'] == 0:
                strata['absent'].append(r)
            else:
                strata[r['family_gap']].append(r)

        def stratum_block(bucket):
            if not bucket:
                return None
            return {
                'n': len(bucket),
                'label_match_rate': float(np.mean([r['label_match'] for r in bucket])),
                'behavior_preserved_rate': float(np.mean([r['behavior_preserved'] for r in bucket])),
                'per_class_pred': dict(Counter(r['tgt_pred'] for r in bucket)),
            }

        by_stratum = {}
        for gap in ['near', 'absent', 'moderate', 'extreme']:
            block = stratum_block(strata.get(gap, []))
            if block is not None:
                by_stratum[gap] = block
        by_stratum['all'] = stratum_block(valid) or {
            'n': 0, 'label_match_rate': 0.0, 'behavior_preserved_rate': 0.0,
            'per_class_pred': {},
        }

        all_results[method] = {
            'n_valid': len(valid),
            'n_total': len(per_pair),
            'overall_label_match_rate': by_stratum['all']['label_match_rate'],
            'overall_behavior_preserved_rate': by_stratum['all']['behavior_preserved_rate'],
            'by_stratum': by_stratum,
            'per_class_pred_overall': dict(Counter(r['tgt_pred'] for r in valid)),
            'per_pair': per_pair,
        }

    # Source-side prediction distribution for reference
    src_dist = dict(Counter(v for v in src_pred_by_pair.values() if v is not None))

    OUT.write_text(json.dumps({
        'eval_set': str(EVAL_PAIRS),
        'n_pairs': len(pairs),
        'classifier': str(CLF_CKPT),
        'classifier_arch': clf.arch,
        'classifier_val_acc': float(getattr(clf, 'val_acc', 0.0)) if hasattr(clf, 'val_acc') else None,
        'action_classes': list(ACTION_CLASSES),
        'source_pred_distribution': src_dist,
        'methods': all_results,
    }, indent=2))

    # Pretty-print
    print("\n" + "=" * 110, flush=True)
    print(f"UNIFIED COMPARISON v2 (external_classifier_v2) — saved to {OUT}")
    print("=" * 110, flush=True)
    print(f"{'Method':<24} {'valid':>5} {'lbl all':>8} {'beh all':>8} {'lbl near':>9} {'lbl abs':>8} {'lbl mod':>8} {'lbl ext':>8}")
    for method, s in all_results.items():
        strat = s.get('by_stratum', {})

        def g(key, field):
            b = strat.get(key, {})
            v = b.get(field) if b else None
            return f"{v:.3f}" if isinstance(v, (int, float)) else '—'
        print(f"{method:<24} {s['n_valid']:>5} "
              f"{s['overall_label_match_rate']:>8.3f} {s['overall_behavior_preserved_rate']:>8.3f} "
              f"{g('near', 'label_match_rate'):>9} {g('absent', 'label_match_rate'):>8} "
              f"{g('moderate', 'label_match_rate'):>8} {g('extreme', 'label_match_rate'):>8}")

    # Per-class prediction counts
    print("\nPer-class prediction counts (overall across 30 pairs):", flush=True)
    classes = list(ACTION_CLASSES)
    header = f"{'Method':<24} " + ' '.join(f"{c[:5]:>5}" for c in classes)
    print(header)
    for method, s in all_results.items():
        counts = s['per_class_pred_overall']
        row = f"{method:<24} " + ' '.join(f"{counts.get(c, 0):>5}" for c in classes)
        print(row)

    print(f"\nSource distribution: {src_dist}", flush=True)
    print(f"\nSaved: {OUT}", flush=True)


if __name__ == '__main__':
    main()
