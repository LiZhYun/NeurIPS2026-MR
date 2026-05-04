"""Round 3 of auto-review: supervised CFM on z-score DTW + cleaned adversarial.

Changes vs v1:
1. Supervised CFM sampled via chunked inference on ALL 200 pairs (windows 40)
2. Cleaned adversarial — use coarse action CLUSTERS, not just disjoint single actions
3. Report z-score DTW for every method against every baseline

Output: eval/results/k_compare/metric_stress_test_v2.json
"""
from __future__ import annotations
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.benchmark_paired.metrics.end_effector_dtw import end_effector_dtw
from eval.benchmark_paired.metrics.end_effector_dtw_normalized import (
    end_effector_dtw_normalized,
)
from eval.benchmark_paired.metrics.contact_timing_f1 import contact_timing_f1
from model.skel_blind.cfm_model import SLOT_COUNT, CHANNEL_COUNT
from model.skel_blind.encoder import encode_motion_to_invariant
from sample.generate_cfm import load_model, sample_euler

DATA_ROOT = PROJECT_ROOT / 'dataset/truebones/zoo/truebones_processed'
PAIRS_DIR = PROJECT_ROOT / 'eval/benchmark_paired/pairs'
K_RETRIEVE_ONLY_DIR = PROJECT_ROOT / 'eval/results/k_compare/K_retrieve_only_200pair'
K_FRAME_NN_DIR = PROJECT_ROOT / 'eval/results/k_compare/K_frame_nn_200pair'
SUPERVISED_CKPT = PROJECT_ROOT / 'save/cfm_supervised/ckpt_final.pt'
MOTION_DIR = DATA_ROOT / 'motions'

# Coarse action clusters — adversarial picks a clip from a DIFFERENT cluster
ACTION_CLUSTERS = {
    'locomotion': {'walk', 'walkforward', 'slowwalk', 'run', 'running', 'runloop', 'trot',
                   'backup', 'turnleft', 'turnright', 'rearing', 'land', 'landing', 'takeoff'},
    'combat': {'attack', 'attackright', 'attackleft', 'bite', 'fight', 'hit', 'roar',
               'growl', 'jumpbite', 'hoofscrape', 'throw'},
    'idle': {'idle', 'idleloop', 'slowidle', 'rest', 'restless', 'sit', 'shake', 'sniff', 'eat', 'yawn', 'stand'},
    'death': {'die', 'dieloop', 'death', 'fall', 'knockedback'},
    'fly': {'fly', 'flyloop', 'slowfly', 'glide'},
    'jump': {'jump', 'jumpforward', 'rise', 'getup', 'getupagain', 'getupfromside'},
    'agitated': {'agitated', 'scared', 'tailwhip'},
}

def action_to_cluster(action):
    a = action.lower()
    for cluster, members in ACTION_CLUSTERS.items():
        if a in members:
            return cluster
    return 'other'


def bootstrap_ci(values, n_boot=1000, seed=42):
    rng = np.random.RandomState(seed)
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return (0.0, 0.0, 0.0)
    means = np.array([rng.choice(arr, size=n, replace=True).mean() for _ in range(n_boot)])
    return (float(np.percentile(means, 2.5)), float(arr.mean()), float(np.percentile(means, 97.5)))


def win_rate_ci(a, b, lower_is_better=True, n_boot=1000, seed=42):
    rng = np.random.RandomState(seed)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    wins = (a < b) if lower_is_better else (a > b)
    wins = wins.astype(np.float64)
    rates = np.array([rng.choice(wins, size=len(wins), replace=True).mean() for _ in range(n_boot)])
    return (float(np.percentile(rates, 2.5)), float(wins.mean()), float(np.percentile(rates, 97.5)))


def make_skel_cond(cond_dict, name):
    return {
        'joints_names': cond_dict[name]['joints_names'],
        'parents': cond_dict[name]['parents'],
        'object_type': name,
    }


def parse_action_coarse(fname):
    import re
    parts = fname.split('___')
    if len(parts) >= 2:
        raw = parts[1].rsplit('_', 1)[0].lower()
        return re.sub(r'\d+$', '', raw) or raw
    return fname.lower()


def list_tgt_clips(skel):
    return sorted([f for f in os.listdir(MOTION_DIR)
                   if f.startswith(skel + '___') and f.endswith('.npy')])


@torch.no_grad()
def sample_supervised_cfm(model, inv_a, window, device, n_steps=30, cfg_weight=2.0):
    """Sample supervised CFM on inv_a (source rep). Chunked inference for T > window."""
    T = inv_a.shape[0]
    if T <= window:
        inv_padded = np.zeros((window, SLOT_COUNT, CHANNEL_COUNT), dtype=np.float32)
        inv_padded[:T] = inv_a
        z_src = torch.from_numpy(inv_padded).float().unsqueeze(0).to(device)
        out = sample_euler(model, z_src, n_steps=n_steps, cfg_weight=cfg_weight)
        return out[0].cpu().numpy()[:T]
    # Chunked
    chunks = []
    for start in range(0, T, window):
        end = min(start + window, T)
        chunk = inv_a[start:end]
        inv_padded = np.zeros((window, SLOT_COUNT, CHANNEL_COUNT), dtype=np.float32)
        inv_padded[:chunk.shape[0]] = chunk
        z_src = torch.from_numpy(inv_padded).float().unsqueeze(0).to(device)
        out = sample_euler(model, z_src, n_steps=n_steps, cfg_weight=cfg_weight)
        chunks.append(out[0].cpu().numpy()[:chunk.shape[0]])
    return np.concatenate(chunks, axis=0)


def main():
    print("Loading cond_dict...")
    cond_dict = np.load(DATA_ROOT / 'cond.npy', allow_pickle=True).item()

    with open(PAIRS_DIR / 'manifest.json') as f:
        manifest = json.load(f)
    pairs = manifest['pairs']
    print(f"Loaded {len(pairs)} pairs")

    print(f"Loading supervised CFM from {SUPERVISED_CKPT}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sup_model, sup_args = load_model(str(SUPERVISED_CKPT), device)
    window = sup_args.get('window', 40)

    tgt_clips = {}
    rng = np.random.RandomState(42)

    results = []
    for i, p in enumerate(pairs):
        pid = p['pair_id']
        skel_b = p['skel_b']
        action = p['action']
        gt_file = p['file_b']
        gt_cluster = action_to_cluster(action)

        pair_data = np.load(PAIRS_DIR / p['pair_file'])
        inv_a = pair_data['inv_a']
        inv_b = pair_data['inv_b']
        T_b = inv_b.shape[0]
        skel_cond = make_skel_cond(cond_dict, skel_b)

        # Target pool
        if skel_b not in tgt_clips:
            tgt_clips[skel_b] = list_tgt_clips(skel_b)
        pool = [f for f in tgt_clips[skel_b] if f != gt_file]
        if not pool:
            continue

        # Random
        rand_fname = pool[rng.randint(0, len(pool))]
        rand_motion = np.load(MOTION_DIR / rand_fname)[:T_b]

        # Adversarial: DIFFERENT CLUSTER
        adv_candidates = [f for f in pool
                          if action_to_cluster(parse_action_coarse(f)) != gt_cluster
                          and action_to_cluster(parse_action_coarse(f)) != 'other']
        if not adv_candidates:
            continue  # skip — no valid adversarial
        adv_fname = adv_candidates[rng.randint(0, len(adv_candidates))]
        adv_motion = np.load(MOTION_DIR / adv_fname)[:T_b]
        adv_cluster = action_to_cluster(parse_action_coarse(adv_fname))

        # K_retrieve
        kr_path = K_RETRIEVE_ONLY_DIR / f'pair_{pid:04d}.npy'
        knn_path = K_FRAME_NN_DIR / f'pair_{pid:04d}.npy'
        if not kr_path.exists() or not knn_path.exists():
            continue
        kr_motion = np.load(kr_path)[:T_b]
        knn_motion = np.load(knn_path)[:T_b]

        try:
            inv_kr = encode_motion_to_invariant(kr_motion, skel_cond)
            inv_knn = encode_motion_to_invariant(knn_motion, skel_cond)
            inv_rand = encode_motion_to_invariant(rand_motion, skel_cond)
            inv_adv = encode_motion_to_invariant(adv_motion, skel_cond)

            # Supervised CFM (chunked)
            inv_sup = sample_supervised_cfm(sup_model, inv_a, window, device)[:T_b]

            def all_metrics(inv_x):
                return {
                    'dtw_raw': float(end_effector_dtw(inv_x, inv_b)),
                    'dtw_zscore': float(end_effector_dtw_normalized(inv_x, inv_b, 'zscore')),
                    'dtw_scale': float(end_effector_dtw_normalized(inv_x, inv_b, 'scale')),
                    'f1': float(contact_timing_f1(inv_x, inv_b)),
                }

            rec = {
                'pair_id': pid, 'action': action, 'gt_cluster': gt_cluster,
                'adv_cluster': adv_cluster, 'target_skel': skel_b,
                'zero_training': all_metrics(inv_a[:T_b]),
                'k_retrieve': all_metrics(inv_kr),
                'k_frame_nn': all_metrics(inv_knn),
                'random': all_metrics(inv_rand),
                'adversarial': all_metrics(inv_adv),
                'supervised_cfm': all_metrics(inv_sup),
            }
            results.append(rec)

            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1}/{len(pairs)}] ZT_zscore={rec['zero_training']['dtw_zscore']:.3f} "
                      f"SUP_zscore={rec['supervised_cfm']['dtw_zscore']:.3f} "
                      f"K_zscore={rec['k_retrieve']['dtw_zscore']:.3f} "
                      f"ADV_zscore={rec['adversarial']['dtw_zscore']:.3f}")

        except Exception as e:
            print(f"  FAILED pair {pid}: {e}")

    print(f"\n{'='*80}")
    print(f"STRESS TEST v2: supervised + cleaned adversarial (cluster-disjoint)")
    print(f"{'='*80}")
    print(f"n = {len(results)}")

    methods = ['zero_training', 'supervised_cfm', 'k_retrieve', 'k_frame_nn', 'adversarial', 'random']
    metrics = ['dtw_raw', 'dtw_zscore', 'dtw_scale', 'f1']

    for metric in metrics:
        print(f"\n--- {metric} ({'↓' if 'dtw' in metric else '↑'}) ---")
        for method in methods:
            vals = [r[method][metric] for r in results]
            ci = bootstrap_ci(vals)
            print(f"  {method:18s}: {ci[1]:.4f} [{ci[0]:.4f}, {ci[2]:.4f}]")

    print(f"\n--- PAIRED WINS — K_frame_nn (NEW METHOD) vs each baseline ---")
    for target_method in ['k_frame_nn', 'k_retrieve', 'zero_training']:
        print(f"\n  [{target_method} is method A]")
        for baseline in ['supervised_cfm', 'adversarial', 'random', 'zero_training', 'k_retrieve']:
            if baseline == target_method:
                continue
            for metric in metrics:
                lower = 'dtw' in metric
                direction = '↓' if lower else '↑'
                a = [r[target_method][metric] for r in results]
                b = [r[baseline][metric] for r in results]
                wr = win_rate_ci(a, b, lower_is_better=lower)
                print(f"    {target_method:15s} vs {baseline:14s} {metric:10s} ({direction}): "
                      f"win={wr[1]*100:.1f}% [{wr[0]*100:.1f}, {wr[2]*100:.1f}]")

    out_path = PROJECT_ROOT / 'eval/results/k_compare/metric_stress_test_v2.json'
    with open(out_path, 'w') as f:
        json.dump({'n': len(results), 'per_pair': results}, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
