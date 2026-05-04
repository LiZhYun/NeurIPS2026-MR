"""Cross-skeleton action retrieval — orthogonal validation per reviewer.

For each query clip on skeleton A, retrieve top-K clips from a DIFFERENT
skeleton B by invariant-rep similarity. Compute action-label retrieval accuracy.

This is independent of our paired-DTW benchmark and tests whether the
invariant rep preserves action semantics across skeletons (the reviewer's
orthogonal validator).

Methods compared:
  - invariant_rep_zdtw: z-score DTW on invariant rep (our primary)
  - invariant_rep_raw:  raw DTW on invariant rep
  - q_signature: cosine on 20-dim Q signature (K_retrieve)
  - random: random ranking

Output: eval/results/cross_skel_retrieval.json
"""
from __future__ import annotations
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.benchmark_paired.metrics.end_effector_dtw import end_effector_dtw
from eval.benchmark_paired.metrics.end_effector_dtw_normalized import end_effector_dtw_normalized
from model.skel_blind.invariant_dataset import INVARIANT_DIR, TEST_SKELETONS

CLIP_META = PROJECT_ROOT / 'eval/results/effect_cache/clip_metadata.json'
Q_CACHE = PROJECT_ROOT / 'idea-stage/quotient_cache.npz'

# Action labels to evaluate (skip 'other' which is a catchall, and rare classes)
EVAL_LABELS = {'idle', 'attack', 'die', 'run', 'walk', 'fly', 'turn', 'jump', 'eat', 'getup'}


def load_data():
    with open(CLIP_META) as f:
        meta = json.load(f)
    # Filter to clips with eval labels
    valid = [m for m in meta if m.get('coarse_label') in EVAL_LABELS]
    print(f"Total clips: {len(meta)}, with eval labels: {len(valid)}")

    # Load invariant manifest + cached arrays
    with open(os.path.join(INVARIANT_DIR, 'manifest.json')) as f:
        inv_manifest = json.load(f)

    # Cache invariant reps per skeleton
    inv_cache = {}
    for skel in inv_manifest['skeletons']:
        npz_path = os.path.join(INVARIANT_DIR, f'{skel}.npz')
        inv_cache[skel] = np.load(npz_path, allow_pickle=True)

    # Load Q signatures from quotient cache
    qc = np.load(Q_CACHE, allow_pickle=True)
    qmeta = list(qc['meta'])
    qfname_to_idx = {m['fname']: i for i, m in enumerate(qmeta)}

    # Build per-clip dicts
    clips = []
    for m in valid:
        skel = m['skeleton']
        fname = m['fname']
        stem = os.path.splitext(fname)[0]
        if stem not in inv_cache.get(skel, {}):
            continue
        inv_rep = inv_cache[skel][stem]
        # Q signature
        q_sig = None
        if fname in qfname_to_idx:
            from eval.pilot_Q_experiments import q_signature
            qidx = qfname_to_idx[fname]
            q = {
                'com_path': qc['com_path'][qidx],
                'heading_vel': qc['heading_vel'][qidx],
                'contact_sched': qc['contact_sched'][qidx],
                'cadence': float(qc['cadence'][qidx]),
                'limb_usage': qc['limb_usage'][qidx],
            }
            q_sig = q_signature(q)
        clips.append({
            'fname': fname, 'skel': skel, 'label': m['coarse_label'],
            'inv': inv_rep, 'q_sig': q_sig,
        })
    return clips


def cross_skel_retrieval(clips, distance_fn, query_filter=None, max_query_per_skel=None,
                         top_k=(1, 5), label='unnamed'):
    """For each query, retrieve top-K clips from each OTHER skeleton.
    Returns: dict with hit rates per top-K cutoff."""
    rng = np.random.RandomState(42)
    queries = [c for c in clips if query_filter is None or query_filter(c)]
    if max_query_per_skel:
        per_skel = defaultdict(list)
        for c in queries:
            per_skel[c['skel']].append(c)
        sampled = []
        for skel, qs in per_skel.items():
            rng.shuffle(qs)
            sampled.extend(qs[:max_query_per_skel])
        queries = sampled

    # Build per-skeleton DBs
    by_skel = defaultdict(list)
    for c in clips:
        by_skel[c['skel']].append(c)

    # Stats
    n_queries = 0
    hits_at_k = {k: 0 for k in top_k}
    total_skel_pairs = 0

    t0 = time.time()
    for qi, q in enumerate(queries):
        for tgt_skel, db_clips in by_skel.items():
            if tgt_skel == q['skel']:
                continue
            # Compute distances
            scores = []
            for c in db_clips:
                d = distance_fn(q, c)
                if d is None:
                    continue
                scores.append((d, c['label']))
            if not scores:
                continue
            scores.sort()
            for k in top_k:
                top_labels = [s[1] for s in scores[:k]]
                if q['label'] in top_labels:
                    hits_at_k[k] += 1
            total_skel_pairs += 1
        n_queries += 1
        if (qi + 1) % 20 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (qi + 1) * (len(queries) - qi - 1)
            print(f"  [{label}] [{qi+1}/{len(queries)}] elapsed {elapsed:.0f}s, ETA {eta:.0f}s")

    return {
        'method': label,
        'n_queries': n_queries,
        'n_query_skel_pairs': total_skel_pairs,
        'top_1_acc': hits_at_k[1] / max(total_skel_pairs, 1),
        'top_5_acc': hits_at_k[5] / max(total_skel_pairs, 1),
        'time_sec': time.time() - t0,
    }


def main():
    print("Loading data...")
    clips = load_data()
    print(f"Loaded {len(clips)} clips")
    n_per_skel = defaultdict(int)
    for c in clips:
        n_per_skel[c['skel']] += 1
    print(f"Skeletons: {len(n_per_skel)}, range {min(n_per_skel.values())}–{max(n_per_skel.values())} clips")

    # 1. Z-score DTW on invariant rep (our primary metric)
    def zdtw(q, c):
        try:
            return end_effector_dtw_normalized(q['inv'], c['inv'], 'zscore')
        except Exception:
            return None

    # 2. Raw DTW on invariant rep
    def raw_dtw(q, c):
        try:
            return end_effector_dtw(q['inv'], c['inv'])
        except Exception:
            return None

    # 3. Q-signature cosine
    def q_cosine(q, c):
        if q['q_sig'] is None or c['q_sig'] is None:
            return None
        a, b = q['q_sig'], c['q_sig']
        return float(-(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    # 4. Random
    rng = np.random.RandomState(42)
    def random_dist(q, c):
        return rng.random()

    # Sample queries: limit per skeleton to keep it tractable
    MAX_PER_SKEL = 5  # ~ 70 skeletons * 5 = 350 queries
    print(f"\nMax {MAX_PER_SKEL} queries per skeleton (estimated total queries: {sum(min(MAX_PER_SKEL, n) for n in n_per_skel.values())})")

    results = {}
    for name, dist_fn in [
        ('q_cosine', q_cosine),  # fastest first as smoke test
        ('raw_dtw', raw_dtw),
        ('zdtw', zdtw),
        ('random', random_dist),
    ]:
        print(f"\n=== Method: {name} ===")
        r = cross_skel_retrieval(clips, dist_fn, max_query_per_skel=MAX_PER_SKEL, label=name)
        results[name] = r
        print(f"  Top-1: {r['top_1_acc']*100:.2f}%  Top-5: {r['top_5_acc']*100:.2f}%  "
              f"({r['n_query_skel_pairs']} query-skel pairs, {r['time_sec']:.0f}s)")

    print(f"\n{'='*60}")
    print(f"CROSS-SKELETON ACTION RETRIEVAL (orthogonal validation)")
    print(f"{'='*60}")
    print(f"{'Method':18s} {'Top-1':>10s} {'Top-5':>10s}")
    for name, r in results.items():
        print(f"{name:18s} {r['top_1_acc']*100:>9.2f}% {r['top_5_acc']*100:>9.2f}%")

    # Save
    out = PROJECT_ROOT / 'eval/results/cross_skel_retrieval.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
