"""Compute action-classifier accuracy for K outputs (and any other method directory)
on the 30 canonical eval pairs. Writes a unified comparison JSON.

Protocol: for each pair, load the method's retargeted motion, run the external
classifier on it, compare predicted label to source_label.
"""
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(str(PROJECT_ROOT))
sys.path.insert(0, str(ROOT))

EVAL_PAIRS = ROOT / 'idea-stage/eval_pairs.json'
COMPARE_DIR = ROOT / 'eval/results/k_compare'
CLF_CKPT = ROOT / 'save/external_classifier.pt'
OUT = ROOT / 'idea-stage/unified_method_comparison.json'

METHODS = ['K', 'label_random', 'psi_retrieval', 'q_retrieval', 'q_label_retrieval',
           'motion2motion', 'A_full', 'A_no_action']

DATASET_DIR = ROOT / 'dataset/truebones/zoo/truebones_processed'


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    from eval.external_classifier import ActionClassifier, extract_classifier_features, ACTION_CLASSES
    from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np

    state = torch.load(str(CLF_CKPT), map_location='cpu', weights_only=False)
    clf = ActionClassifier().to(device).eval()
    clf.load_state_dict(state['model'])

    cond = np.load(DATASET_DIR / 'cond.npy', allow_pickle=True).item()

    with open(EVAL_PAIRS) as f:
        eval_set = json.load(f)
    pairs = eval_set['pairs']

    @torch.no_grad()
    def classify_motion(motion_13dim, skel):
        """motion_13dim: [T, J, 13] → predicted action class name, or None if infeasible."""
        J_skel = cond[skel]['offsets'].shape[0]
        m = motion_13dim
        if m.shape[1] > J_skel:
            m = m[:, :J_skel]
        # Handle normalized vs denormalized: some agents saved z-scored, some denormalized.
        # Heuristic: if absolute values are > 5, assume denormalized; else normalized.
        if np.abs(m).max() < 5:
            # Normalized — denormalize using cond stats
            mean = cond[skel]['mean'][:J_skel]
            std = cond[skel]['std'][:J_skel]
            m = m.astype(np.float32) * std + mean
        try:
            positions = recover_from_bvh_ric_np(m.astype(np.float32))
        except Exception as e:
            return None, str(e)
        parents = cond[skel]['parents'][:J_skel]
        feats = extract_classifier_features(positions, parents)
        if feats is None:
            return None, 'feats_none'
        x = torch.tensor(feats, dtype=torch.float32, device=device).unsqueeze(0)
        logits = clf(x)
        pred_idx = int(logits.argmax(-1).item())
        return ACTION_CLASSES[pred_idx], None

    # Source classifier predictions (ground truth anchor)
    print("\nComputing source classifier predictions...")
    src_pred_by_pair = {}
    src_stride_cache = {}
    for p in pairs:
        src_fname = p['source_fname']
        if src_fname in src_stride_cache:
            src_pred_by_pair[p['pair_id']] = src_stride_cache[src_fname]
            continue
        m = np.load(DATASET_DIR / 'motions' / src_fname)
        pred, _ = classify_motion(m, p['source_skel'])
        src_stride_cache[src_fname] = pred
        src_pred_by_pair[p['pair_id']] = pred

    # For each method: per-pair classification
    all_results = {}
    for method in METHODS:
        method_dir = COMPARE_DIR / method
        if not method_dir.exists():
            print(f"  skip {method}: no dir")
            continue
        per_pair = []
        for p in pairs:
            pid = p['pair_id']
            # Find the motion file for this method + pair_id
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
        print(f"  {method}: valid {len(valid)}/{len(per_pair)}")

        # Stratify: absent stratum takes priority (any pair with support_same_label == 0),
        # otherwise family_gap (near / moderate / extreme).
        strata = defaultdict(list)
        for r in valid:
            if r['support_same_label'] == 0:
                strata['absent'].append(r)
            else:
                strata[r['family_gap']].append(r)
        summary = {
            'n_valid': len(valid),
            'n_total': len(per_pair),
            'overall_label_match_rate': float(np.mean([r['label_match'] for r in valid])) if valid else 0,
            'overall_behavior_preserved_rate': float(np.mean([r['behavior_preserved'] for r in valid])) if valid else 0,
            'by_stratum': {}
        }
        for gap in ['near', 'absent', 'moderate', 'extreme']:
            bucket = strata.get(gap, [])
            if bucket:
                summary['by_stratum'][gap] = {
                    'n': len(bucket),
                    'label_match_rate': float(np.mean([r['label_match'] for r in bucket])),
                    'behavior_preserved_rate': float(np.mean([r['behavior_preserved'] for r in bucket])),
                }
        all_results[method] = summary

    OUT.write_text(json.dumps({
        'eval_set': str(EVAL_PAIRS),
        'n_pairs': len(pairs),
        'methods': all_results,
    }, indent=2))

    # Pretty-print comparison
    print("\n" + "=" * 100)
    print(f"UNIFIED COMPARISON (action-class accuracy via external classifier)")
    print("=" * 100)
    print(f"{'Method':<22} {'overall lbl':>12} {'overall beh':>12} {'near lbl':>10} {'absent lbl':>10} {'mod lbl':>10} {'ext lbl':>10}")
    for method, s in all_results.items():
        strat = s.get('by_stratum', {})
        near = strat.get('near', {}).get('label_match_rate', None)
        absent = strat.get('absent', {}).get('label_match_rate', None)
        mod = strat.get('moderate', {}).get('label_match_rate', None)
        ext = strat.get('extreme', {}).get('label_match_rate', None)
        near_s = f"{near:.3f}" if near is not None else '—'
        absent_s = f"{absent:.3f}" if absent is not None else '—'
        mod_s = f"{mod:.3f}" if mod is not None else '—'
        ext_s = f"{ext:.3f}" if ext is not None else '—'
        print(f"{method:<22} {s['overall_label_match_rate']:>12.3f} {s['overall_behavior_preserved_rate']:>12.3f} {near_s:>10} {absent_s:>10} {mod_s:>10} {ext_s:>10}")

    print(f"\nSaved: {OUT}")


if __name__ == '__main__':
    main()
