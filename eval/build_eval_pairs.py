"""Build the canonical 30-pair eval set used by Idea K + baselines + competitors.

Pairs are (source_clip_fname, source_skel, target_skel). Stratified by support regime and
morphology gap so that every method is measured on the same pairs.

Output: /home//Codes/Anytop/idea-stage/eval_pairs.json

Strata (target = 30 pairs total):
  - 10 support-present near (same morphology family)
  - 10 support-absent (rare source action on non-compatible target)
  - 5  cross-family moderate (mammal↔bird, reptile↔insect, etc.)
  - 5  extreme (snake↔biped, 4-leg↔8-leg, ground↔flyer)
"""
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import json
from collections import Counter, defaultdict
from pathlib import Path
import random

import numpy as np

ROOT = Path(str(PROJECT_ROOT))
META_PATH = ROOT / 'eval/results/effect_cache/clip_metadata.json'
CONTACT_GROUPS = ROOT / 'eval/quotient_assets/contact_groups.json'
OUT = ROOT / 'idea-stage/eval_pairs.json'
SEED = 42

# Morphology families (coarse, for stratification only)
FAMILY = {
    'mammal_quad': ['Horse', 'Bear', 'BrownBear', 'Cat', 'Lion', 'Lynx', 'Leapord', 'Jaguar',
                    'Wolf', 'Fox', 'Coyote', 'Hound', 'SabreToothTiger', 'PolarBear', 'PolarBearB',
                    'Hamster', 'Rat', 'Deer', 'Gazelle', 'Camel', 'Buffalo', 'Hippopotamus',
                    'Mammoth', 'Elephant', 'Goat', 'Rhino', 'Skunk', 'Puppy', 'SandMouse',
                    'Raindeer'],
    'reptile': ['Alligator', 'Crocodile', 'Comodoa', 'Turtle'],
    'dino': ['Trex', 'Tyranno', 'Raptor', 'Raptor2', 'Raptor3', 'Stego', 'Tricera'],
    'snake': ['Anaconda', 'KingCobra'],
    'bird_ground': ['Chicken', 'Ostrich', 'Flamingo'],
    'bird_fly': ['Bird', 'Buzzard', 'Eagle', 'Parrot', 'Parrot2', 'Pigeon'],
    'insect_6': ['Ant', 'FireAnt', 'Giantbee', 'Cricket', 'Roach', 'Scorpion', 'Scorpion-2'],
    'insect_8': ['Spider', 'SpiderG', 'Centipede'],
    'crustacean': ['Crab', 'HermitCrab', 'Isopetra'],
    'flying_mammal': ['Bat'],
    'dragon': ['Dragon', 'Pteranodon'],
    'other': ['Monkey', 'Tukan', 'Pirrana'],
}
SKEL_TO_FAMILY = {s: f for f, ss in FAMILY.items() for s in ss}


def pair_family_gap(src_skel, tgt_skel):
    sf = SKEL_TO_FAMILY.get(src_skel, 'other')
    tf = SKEL_TO_FAMILY.get(tgt_skel, 'other')
    if sf == tf:
        return 'near'
    # Moderate vs extreme: manual judgment — snake↔biped, 4↔8 leg, etc.
    extreme_pairs = [
        ('snake', 'mammal_quad'), ('snake', 'insect_8'), ('snake', 'bird_fly'),
        ('insect_8', 'bird_fly'), ('insect_8', 'flying_mammal'),
        ('mammal_quad', 'insect_6'), ('mammal_quad', 'crustacean'),
        ('bird_fly', 'reptile'), ('bird_fly', 'snake'),
        ('crustacean', 'bird_ground'),
    ]
    if (sf, tf) in extreme_pairs or (tf, sf) in extreme_pairs:
        return 'extreme'
    return 'moderate'


def main():
    rng = np.random.default_rng(SEED)
    with open(META_PATH) as f:
        meta = json.load(f)
    with open(CONTACT_GROUPS) as f:
        groups = json.load(f)

    # Skeletons with contact_groups authored (drop the 3 unresolved)
    authored = [k for k in groups.keys() if not k.startswith('_') and k != '_unresolved']
    unresolved = groups.get('_unresolved', [])
    candidate_skels = [s for s in authored if s not in unresolved]

    # Per-skel label support
    label_counts_by_skel = defaultdict(Counter)
    skel_to_clips = defaultdict(list)
    for i, m in enumerate(meta):
        skel_to_clips[m['skeleton']].append((i, m))
        label_counts_by_skel[m['skeleton']][m['coarse_label']] += 1

    val_clips = [(i, m) for i, m in enumerate(meta) if m['split'] == 'val'
                 and m['skeleton'] in candidate_skels]

    pairs_near_present = []
    pairs_absent = []
    pairs_moderate = []
    pairs_extreme = []

    rng_state = random.Random(SEED)
    rng_state.shuffle(val_clips)

    for clip_i, src_m in val_clips:
        if len(pairs_near_present) >= 10 and len(pairs_absent) >= 10 and \
           len(pairs_moderate) >= 5 and len(pairs_extreme) >= 5:
            break
        src_skel = src_m['skeleton']
        src_label = src_m['coarse_label']
        # Try several random targets for this source
        other_skels = [s for s in candidate_skels if s != src_skel]
        rng_state.shuffle(other_skels)
        for tgt_skel in other_skels[:15]:
            support = label_counts_by_skel[tgt_skel][src_label]
            gap = pair_family_gap(src_skel, tgt_skel)
            entry = {
                'source_fname': src_m['fname'],
                'source_skel': src_skel,
                'source_label': src_label,
                'target_skel': tgt_skel,
                'support_same_label': int(support),
                'family_gap': gap,
            }
            if gap == 'near' and support >= 1 and len(pairs_near_present) < 10:
                pairs_near_present.append(entry); break
            elif support == 0 and len(pairs_absent) < 10:
                pairs_absent.append(entry); break
            elif gap == 'moderate' and len(pairs_moderate) < 5:
                pairs_moderate.append(entry); break
            elif gap == 'extreme' and len(pairs_extreme) < 5:
                pairs_extreme.append(entry); break

    pairs = pairs_near_present + pairs_absent + pairs_moderate + pairs_extreme
    for idx, p in enumerate(pairs):
        p['pair_id'] = idx
    summary = {
        'seed': SEED,
        'n_pairs': len(pairs),
        'stratification': {
            'near_present': len(pairs_near_present),
            'absent': len(pairs_absent),
            'moderate': len(pairs_moderate),
            'extreme': len(pairs_extreme),
        },
        'pairs': pairs,
    }
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"Saved {len(pairs)} pairs to {OUT}")
    print(f"Stratification: {summary['stratification']}")
    for p in pairs[:3]:
        print(f"  {p}")


if __name__ == '__main__':
    main()
