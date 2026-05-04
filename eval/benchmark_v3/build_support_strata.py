"""Tag each v3 query as support-present or support-absent.

Per Codex review (2026-04-23): v3 conflates regime types. The oral story requires
distinguishing queries where the target skel HAS a clip with the source's exact
action (support-present, ~70%) from queries where it does NOT (support-absent, ~30%).

Output: eval/benchmark_v3/queries/fold_{F}/support_strata.json
        — one per query: {query_id, support_present (bool), src_action, n_target_skel_with_action}
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CLIP_INDEX_PATH = PROJECT_ROOT / 'eval/benchmark_v3/clip_index.json'
QUERIES_ROOT = PROJECT_ROOT / 'eval/benchmark_v3/queries'


def build_action_lookup():
    """Build {(skel, action) -> count} from the clip index."""
    cidx = json.load(open(CLIP_INDEX_PATH))
    lookup = defaultdict(int)
    for skel, clusters in cidx['index'].items():
        for cluster, clips in clusters.items():
            for clip in clips:
                lookup[(skel, clip['action'])] += 1
    return lookup


def stratify_fold(fold_seed, lookup):
    manifest_path = QUERIES_ROOT / f'fold_{fold_seed}/manifest.json'
    if not manifest_path.exists():
        print(f"SKIP fold {fold_seed}: manifest missing")
        return
    manifest = json.load(open(manifest_path))
    out = []
    n_present = n_absent = 0
    by_split = defaultdict(lambda: {'present': 0, 'absent': 0})
    for q in manifest['queries']:
        skel_b = q['skel_b']
        src_action = q['src_action']
        n_clips = lookup.get((skel_b, src_action), 0)
        sp = (n_clips > 0)
        rec = {
            'query_id': q['query_id'],
            'split': q['split'],
            'src_skel': q['skel_a'],
            'tgt_skel': skel_b,
            'src_action': src_action,
            'cluster': q['cluster'],
            'support_present': sp,
            'n_target_skel_with_action': n_clips,
        }
        out.append(rec)
        if sp:
            n_present += 1
            by_split[q['split']]['present'] += 1
        else:
            n_absent += 1
            by_split[q['split']]['absent'] += 1
    out_path = manifest_path.parent / 'support_strata.json'
    with open(out_path, 'w') as f:
        json.dump({
            'fold': fold_seed,
            'n_queries': len(out),
            'n_support_present': n_present,
            'n_support_absent': n_absent,
            'by_split': dict(by_split),
            'queries': out,
        }, f, indent=2)
    print(f"fold {fold_seed}: {n_present} present, {n_absent} absent")
    print(f"  by_split: {dict(by_split)}")
    print(f"  saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', nargs='+', type=int, default=[42, 43])
    args = parser.parse_args()
    lookup = build_action_lookup()
    print(f"Action lookup: {len(lookup)} (skel, action) entries")
    for fold in args.folds:
        stratify_fold(fold, lookup)


if __name__ == '__main__':
    main()
