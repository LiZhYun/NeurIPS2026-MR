"""Action-classifier scoring for Idea F (idea_F_topo_lora) matching
eval/k_action_accuracy.py's protocol. Writes:
    eval/results/k_compare/idea_F_topo_lora/action_accuracy.json

Also prints a small comparison vs K's reference (0.233 label, 0.267 beh).
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
METHOD_DIR = ROOT / 'eval/results/k_compare/idea_F_topo_lora'
CLF_CKPT = ROOT / 'save/external_classifier.pt'
OUT = METHOD_DIR / 'action_accuracy.json'
DATASET_DIR = ROOT / 'dataset/truebones/zoo/truebones_processed'


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

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
        if feats is None:
            return None, 'feats_none'
        x = torch.tensor(feats, dtype=torch.float32, device=device).unsqueeze(0)
        logits = clf(x)
        pred_idx = int(logits.argmax(-1).item())
        return ACTION_CLASSES[pred_idx], None

    # Source classifier predictions (ground-truth anchor for behavior preservation)
    print('\nComputing source classifier predictions...')
    src_pred_by_pair = {}
    for p in pairs:
        src_fname = p['source_fname']
        m = np.load(DATASET_DIR / 'motions' / src_fname)
        pred, _ = classify_motion(m, p['source_skel'])
        src_pred_by_pair[p['pair_id']] = pred

    per_pair = []
    for p in pairs:
        pid = p['pair_id']
        patterns = [f for f in os.listdir(METHOD_DIR)
                    if f.startswith(f'pair_{pid:02d}_') and f.endswith('.npy')]
        if not patterns:
            per_pair.append({'pair_id': pid, 'error': 'no_output'})
            continue
        motion_path = METHOD_DIR / patterns[0]
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
    strata = defaultdict(list)
    for r in valid:
        if r['support_same_label'] == 0:
            strata['absent'].append(r)
        else:
            strata[r['family_gap']].append(r)
    summary = {
        'method': 'idea_F_topo_lora',
        'n_valid': len(valid),
        'n_total': len(per_pair),
        'overall_label_match_rate': float(np.mean([r['label_match'] for r in valid])) if valid else 0,
        'overall_behavior_preserved_rate': float(np.mean([r['behavior_preserved'] for r in valid])) if valid else 0,
        'by_stratum': {},
        'per_pair': per_pair,
    }
    for gap in ['near', 'absent', 'moderate', 'extreme']:
        bucket = strata.get(gap, [])
        if bucket:
            summary['by_stratum'][gap] = {
                'n': len(bucket),
                'label_match_rate': float(np.mean([r['label_match'] for r in bucket])),
                'behavior_preserved_rate': float(np.mean([r['behavior_preserved'] for r in bucket])),
            }

    OUT.write_text(json.dumps(summary, indent=2))
    print('\n' + '=' * 70)
    print(f"Idea F (topo_lora) action accuracy (n_valid={summary['n_valid']}/{summary['n_total']}):")
    print('=' * 70)
    print(f"  overall label_match:       {summary['overall_label_match_rate']:.3f}   (K ref: 0.233)")
    print(f"  overall behavior_preserved:{summary['overall_behavior_preserved_rate']:.3f}   (K ref: 0.267)")
    for gap in ['near', 'absent', 'moderate', 'extreme']:
        b = summary['by_stratum'].get(gap, {})
        if b:
            print(f"  {gap:<9s} n={b['n']:<2d}  label={b['label_match_rate']:.3f}  "
                  f"beh={b['behavior_preserved_rate']:.3f}")
    print(f"\nSaved: {OUT}")


if __name__ == '__main__':
    main()
