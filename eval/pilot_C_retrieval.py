"""Pilot C — Retrieval-as-Retargeting (Phase 2.6, 2026-04-14)

Reviewer-specified protocol: stratify by target support BEFORE looking at averages.
Strata: (a) source action present vs absent on target, (b) within-present: >=3 clips vs 1.

Operates on the ψ cache (1070 clips × 64 frames × 62 dims) and the clip metadata.
For each (source_clip ∈ val, target_skel ≠ source_skel) pair:
  - retrieve top-k target clips by cosine(ψ_source, ψ_candidate) with matching coarse_label;
    if no matching label on target, fall back to label-free cosine retrieval
  - success if retrieved top-1 clip's coarse_label == source's coarse_label
  - also: report ψ-distance-to-best-available and fraction of pairs with any-support

The pilot does NOT generate full motion sequences — the retrieval question is answered
cheaply in ψ+label space. Motion generation + external-classifier action-accuracy is a
downstream confirmation step that only matters if this retrieval step reveals structure.

Output: idea-stage/pilot_C_results.json
"""
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(str(PROJECT_ROOT))
PSI_PATH = ROOT / 'eval/results/effect_cache/psi_all.npy'
META_PATH = ROOT / 'eval/results/effect_cache/clip_metadata.json'
OUT_PATH = ROOT / 'idea-stage/pilot_C_results.json'

K = 3                 # top-k retrieval
N_SOURCES = 200       # sampled from val split
SEED = 42


def cosine(a, b):
    """a: [n, d], b: [m, d] → [n, m] cosine."""
    an = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    bn = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    return an @ bn.T


def main():
    rng = np.random.default_rng(SEED)
    psi_all = np.load(PSI_PATH)  # [1070, 64, 62]
    with open(META_PATH) as f:
        meta = json.load(f)
    n = len(meta)
    assert psi_all.shape[0] == n, f"psi {psi_all.shape[0]} != meta {n}"

    # Clip-level ψ pooled over time (mean) → [n, 62]
    psi_mean = psi_all.mean(axis=1)

    # Index clips by skeleton and by (skeleton, coarse_label)
    skel_to_idx = defaultdict(list)
    skel_label_to_idx = defaultdict(list)
    label_counts_by_skel = defaultdict(Counter)
    for i, m in enumerate(meta):
        skel_to_idx[m['skeleton']].append(i)
        skel_label_to_idx[(m['skeleton'], m['coarse_label'])].append(i)
        label_counts_by_skel[m['skeleton']][m['coarse_label']] += 1

    skeletons = sorted(skel_to_idx.keys())
    print(f"Skeletons: {len(skeletons)}")
    print(f"Val clips: {sum(1 for m in meta if m['split'] == 'val')}")

    # Pool of source clips from val split
    val_idx = [i for i, m in enumerate(meta) if m['split'] == 'val']
    source_indices = rng.choice(val_idx, size=min(N_SOURCES, len(val_idx)), replace=False)

    # For each source, sample one target skeleton distinct from source_skel
    results = []
    for src_i in source_indices:
        src_meta = meta[src_i]
        src_skel = src_meta['skeleton']
        src_label = src_meta['coarse_label']

        # Candidate target skeletons: all except src_skel
        other_skels = [s for s in skeletons if s != src_skel]
        # Sample one target uniformly
        tgt_skel = rng.choice(other_skels)

        # Determine support on target
        support_same_label = label_counts_by_skel[tgt_skel][src_label]
        support_any = sum(label_counts_by_skel[tgt_skel].values())

        # Retrieve on target
        tgt_pool = skel_to_idx[tgt_skel]
        tgt_pool_labels = [meta[j]['coarse_label'] for j in tgt_pool]

        # Matching-label pool (preferred)
        match_pool = [j for j in tgt_pool if meta[j]['coarse_label'] == src_label]

        if match_pool:
            pool = match_pool
            used_label_match = True
        else:
            # Fallback: pool of all target clips
            pool = tgt_pool
            used_label_match = False

        # Retrieve by cosine(ψ_mean) — top-K
        sims = cosine(psi_mean[src_i:src_i+1], psi_mean[pool])[0]  # [|pool|]
        order = np.argsort(-sims)[:K]
        retrieved_idx = [pool[o] for o in order]
        retrieved_labels = [meta[j]['coarse_label'] for j in retrieved_idx]
        retrieved_sims = sims[order].tolist()

        # Success: top-1 retrieved has same label as source
        top1_label_match = retrieved_labels[0] == src_label
        any_top_k_label_match = any(r == src_label for r in retrieved_labels)

        # Also: what if we ignore label matching entirely — pure ψ retrieval?
        sims_pure = cosine(psi_mean[src_i:src_i+1], psi_mean[tgt_pool])[0]
        order_pure = np.argsort(-sims_pure)[:K]
        pure_top1_label = meta[tgt_pool[order_pure[0]]]['coarse_label']
        pure_top1_label_match = pure_top1_label == src_label

        results.append({
            'src_fname': src_meta['fname'],
            'src_skel': src_skel,
            'src_label': src_label,
            'tgt_skel': tgt_skel,
            'support_same_label': support_same_label,
            'support_any': support_any,
            'used_label_match_pool': used_label_match,
            'retrieved_top_labels': retrieved_labels,
            'retrieved_top_sims': retrieved_sims,
            'top1_label_match': top1_label_match,
            'any_top_k_label_match': any_top_k_label_match,
            'pure_psi_top1_label': pure_top1_label,
            'pure_psi_top1_label_match': pure_top1_label_match,
        })

    # Stratify per reviewer
    def stratify(key):
        buckets = defaultdict(list)
        for r in results:
            buckets[key(r)].append(r)
        return buckets

    # Strat 1: support_same_label present vs absent
    by_support_present = stratify(lambda r: 'present' if r['support_same_label'] > 0 else 'absent')
    # Strat 2: within-present, >=3 vs <3
    by_support_level = stratify(
        lambda r: ('high (>=3)' if r['support_same_label'] >= 3
                   else 'medium (1-2)' if r['support_same_label'] >= 1
                   else 'absent')
    )

    def summarize(bucket):
        if not bucket:
            return {'n': 0}
        return {
            'n': len(bucket),
            'top1_label_match_rate': float(np.mean([r['top1_label_match'] for r in bucket])),
            'any_top_k_label_match_rate': float(np.mean([r['any_top_k_label_match'] for r in bucket])),
            'pure_psi_top1_label_match_rate': float(np.mean([r['pure_psi_top1_label_match'] for r in bucket])),
            'used_label_match_pool_rate': float(np.mean([r['used_label_match_pool_pool'] if 'used_label_match_pool_pool' in r else r['used_label_match_pool'] for r in bucket])),
            'mean_top1_sim': float(np.mean([r['retrieved_top_sims'][0] for r in bucket])),
        }

    summary = {
        'pilot': 'C — Retrieval-as-Retargeting',
        'seed': SEED,
        'n_source_clips': len(source_indices),
        'k': K,
        'overall': summarize(results),
        'stratified_by_support_present': {k: summarize(v) for k, v in by_support_present.items()},
        'stratified_by_support_level': {k: summarize(v) for k, v in by_support_level.items()},
        'per_pair_results': results[:20],  # save first 20 pairs for spot check
    }

    OUT_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\n=== Pilot C summary ===")
    print(f"N source clips: {summary['n_source_clips']}")
    print(f"K (top-k): {K}")
    print(f"\nOverall:")
    for k, v in summary['overall'].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")
    print(f"\nBy support-present strata:")
    for strat, s in summary['stratified_by_support_present'].items():
        print(f"  {strat}:")
        for k, v in s.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.3f}")
            else:
                print(f"    {k}: {v}")
    print(f"\nBy support-level strata:")
    for strat, s in summary['stratified_by_support_level'].items():
        print(f"  {strat}:")
        for k, v in s.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.3f}")
            else:
                print(f"    {k}: {v}")
    print(f"\nSaved: {OUT_PATH}")


if __name__ == '__main__':
    main()
