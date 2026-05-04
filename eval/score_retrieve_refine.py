"""Score the retrieve_refine outputs with the external action classifier
using the same protocol as eval.k_action_accuracy (no modification to existing
files). Writes:
  - eval/results/k_compare/retrieve_refine/action_accuracy.json
  - appends results to idea-stage/unified_method_comparison.json (in-place
    update under methods['retrieve_refine']).

Reused directly: classify_motion logic, source-pred cache, stratification
scheme. We duplicate minimally to avoid importing from k_action_accuracy
(which would execute its METHODS loop).

Usage:
    conda run -n anytop python -m eval.score_retrieve_refine
"""
from __future__ import annotations

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
METHOD_DIR = COMPARE_DIR / 'retrieve_refine'
CLF_CKPT = ROOT / 'save/external_classifier.pt'
UNIFIED = ROOT / 'idea-stage/unified_method_comparison.json'
OUT_LOCAL = METHOD_DIR / 'action_accuracy.json'

DATASET_DIR = ROOT / 'dataset/truebones/zoo/truebones_processed'


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    from eval.external_classifier import (
        ActionClassifier, extract_classifier_features, ACTION_CLASSES)
    from data_loaders.truebones.truebones_utils.motion_process import (
        recover_from_bvh_ric_np)

    state = torch.load(str(CLF_CKPT), map_location='cpu', weights_only=False)
    clf = ActionClassifier().to(device).eval()
    clf.load_state_dict(state['model'])

    cond = np.load(DATASET_DIR / 'cond.npy', allow_pickle=True).item()

    with open(EVAL_PAIRS) as f:
        pairs = json.load(f)['pairs']

    @torch.no_grad()
    def classify_motion(m, skel):
        J_skel = cond[skel]['offsets'].shape[0]
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
        if feats is None:
            return None, 'feats_none'
        x = torch.tensor(feats, dtype=torch.float32, device=device).unsqueeze(0)
        logits = clf(x)
        return ACTION_CLASSES[int(logits.argmax(-1).item())], None

    # Source predictions (cache by fname)
    src_pred_by_pair = {}
    src_cache = {}
    for p in pairs:
        fn = p['source_fname']
        if fn in src_cache:
            src_pred_by_pair[p['pair_id']] = src_cache[fn]
            continue
        m = np.load(DATASET_DIR / 'motions' / fn)
        pred, _ = classify_motion(m, p['source_skel'])
        src_cache[fn] = pred
        src_pred_by_pair[p['pair_id']] = pred

    per_pair = []
    for p in pairs:
        pid = p['pair_id']
        patt = [f for f in os.listdir(METHOD_DIR)
                if f.startswith(f'pair_{pid:02d}_') and f.endswith('.npy')]
        if not patt:
            per_pair.append({'pair_id': pid, 'error': 'no_output'})
            continue
        try:
            motion = np.load(METHOD_DIR / patt[0])
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
    print(f'valid {len(valid)}/{len(per_pair)}')

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
        'by_stratum': {},
    }
    for gap in ['near', 'absent', 'moderate', 'extreme']:
        bucket = strata.get(gap, [])
        if bucket:
            summary['by_stratum'][gap] = {
                'n': len(bucket),
                'label_match_rate': float(np.mean([r['label_match'] for r in bucket])),
                'behavior_preserved_rate': float(np.mean([r['behavior_preserved'] for r in bucket])),
            }

    OUT_LOCAL.write_text(json.dumps(
        {'method': 'retrieve_refine', 'per_pair': per_pair, 'summary': summary},
        indent=2))
    print(f'Saved: {OUT_LOCAL}')

    # Update unified comparison in place (additive).
    if UNIFIED.exists():
        unified = json.loads(UNIFIED.read_text())
        unified.setdefault('methods', {})['retrieve_refine'] = summary
        UNIFIED.write_text(json.dumps(unified, indent=2))
        print(f'Updated {UNIFIED} with retrieve_refine entry.')

    # Pretty-print
    print(f"\nretrieve_refine overall label_match = {summary['overall_label_match_rate']:.3f}  "
          f"behavior_preserved = {summary['overall_behavior_preserved_rate']:.3f}")
    for k, v in summary['by_stratum'].items():
        print(f"  {k:<10} n={v['n']:2d}  label={v['label_match_rate']:.3f}  "
              f"beh={v['behavior_preserved_rate']:.3f}")


if __name__ == '__main__':
    main()
