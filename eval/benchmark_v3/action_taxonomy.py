"""Action taxonomy for v3 benchmark.

Normalizes filename-derived action labels into 11 coarse clusters.
"Other" labeled clips are EXCLUDED from the core benchmark.
"""
from __future__ import annotations
import re
from typing import Optional


# 10 action clusters (drops 'other' catchall, no 'other_kept')
ACTION_CLUSTERS = {
    'locomotion': {
        'walk', 'walkforward', 'walkright', 'walkleft', 'slowwalk', 'walkloop', 'walkcall',
        'run', 'running', 'runloop', 'runs', 'dash',
        'trot', 'trotleft', 'trotright',
        'backup', 'backwards', 'backwalk',
        'turn', 'turnleft', 'turnright',
        'march', 'sneaky',  # stealth locomotion
        'charge',  # locomotion leading to combat
    },
    'combat': {
        'attack', 'attackleft', 'attackright',
        'bite', 'bitehard',
        'fight', 'hit',
        'roar', 'growl',
        'jumpbite',
        'hoofscrape',
        'throw', 'thrown',
        'tailwhip',
        'rearing',
        'strike',
        'headbutt',
        'hit', 'hithead',
        'sting',
        'ready',  # combat ready pose
        'claw', 'clawattack',
    },
    'idle': {
        'idle', 'idleloop',
        'slowidle', 'restless',
        'rest', 'sit', 'stand',
        'sniff', 'eat', 'yawn', 'shake',
        'laydown', 'lie', 'liedown',
        'eatfish', 'grazing', 'drink', 'drinking',
        'cud',  # cud chewing
        'sleepup', 'sleep', 'sleeping',
        'stretchyawnidle', 'lickcleanidle', 'idlepurr', 'idleenergetic', 'idlelaydown',
    },
    'death': {
        'die', 'dieloop', 'death', 'dies', 'deathloop', 'longdeath',
        'fall', 'falling',
        'knockedback', 'knockedbacka',
        'shot', 'dying',
    },
    'fly': {
        'fly', 'flyloop', 'slowfly', 'glide', 'flying', 'flapping',
    },
    'takeoff': {
        'takeoff', 'land', 'landing', 'circleland', 'lander', 'downloop',
    },
    'jump': {
        'jump', 'jumpforward', 'jumping', 'jumps',
        'rise', 'rising',
        'getup', 'getupagain', 'getupfromside',
        'rear',
    },
    'agitated': {
        'agitated', 'scared', 'restless2', 'spin',
    },
    'swim': {
        'swim', 'swims', 'swimming',
    },
    'reaction': {
        'bleat', 'cry', 'vocalize', 'crys', 'yell', 'neigh', 'bark', 'meow',
        'take', 'taken',  # reaction to being taken
    },
}

# Reverse map: action_str → cluster
_ACTION_TO_CLUSTER = {}
for cluster, actions in ACTION_CLUSTERS.items():
    for a in actions:
        _ACTION_TO_CLUSTER[a] = cluster


# Species prefixes that should be stripped from action names
# E.g., "cat_idlepurr" → strip "cat" → "idlepurr"
SPECIES_PREFIXES = {
    'cat', 'dog', 'horse', 'puppy', 'raptor', 'eagle', 'pigeon', 'parrot',
    'deer', 'bear', 'lion', 'tiger', 'hound', 'wolf', 'fox', 'monkey',
    'elephant', 'rhino', 'goat', 'buffalo', 'cow', 'camel', 'gazelle',
    'spider', 'crab', 'snake', 'anaconda', 'alligator', 'turtle', 'frog',
    'fish', 'pirrana', 'shark', 'trex', 'tyranno', 'stego', 'raptor3',
    'chicken', 'ostrich', 'flamingo', 'tukan', 'buzzard', 'pteranodon',
    'bird', 'rat', 'bee', 'giantbee', 'ant', 'fireant', 'roach', 'scorpion',
    'centipede', 'isopetra', 'mantis', 'hermitcrab', 'cricket', 'comodoa',
    'brownbear', 'polarbear', 'polarbearb', 'sabretoothtiger', 'jaguar',
    'coyote', 'raindeer', 'crocodile', 'tricera', 'tukan',
    'kingcobra', 'dragon', 'pegasus', 'bearpolar',
}


def _strip_prefix(s: str, prefix_set=SPECIES_PREFIXES) -> str:
    """Strip species prefix if present. 'cat_idlepurr' → 'idlepurr'."""
    for pfx in sorted(prefix_set, key=len, reverse=True):
        if s.startswith(pfx) and len(s) > len(pfx):
            rest = s[len(pfx):]
            # Require separator or immediate action continuation
            if rest.startswith('_'):
                return rest[1:]
            # Also accept no separator if remaining is valid action name
            if rest[0].isalpha():
                return rest
    return s


def normalize_action(raw: str) -> str:
    """Normalize: 'Attack2' → 'attack', 'Cat_IdlePurr' → 'idlepurr',
    '-_Attack' → 'attack', 'hit_head' → 'hithead'."""
    raw = raw.lower().strip()
    # Strip leading garbage like '-_' or '_'
    raw = re.sub(r'^[-_]+', '', raw)
    # Strip trailing digits and underscores
    stripped = re.sub(r'[_\d]+$', '', raw)
    stripped = stripped if stripped else raw
    # Strip species prefix if present
    stripped = _strip_prefix(stripped)
    # Collapse remaining underscores (hit_head → hithead, long_death → longdeath)
    stripped = stripped.replace('_', '')
    return stripped


def parse_action_from_filename(fname: str) -> str:
    """Extract action label from filename like 'Skel___Attack2_1039.npy'."""
    stem = fname.rsplit('.', 1)[0] if '.' in fname else fname
    if '___' in stem:
        parts = stem.split('___')
        action_part = parts[1]
    elif '_' in stem:
        parts = stem.split('_', 1)
        action_part = parts[1]
    else:
        action_part = stem
    # Drop trailing _NN
    action_part = re.sub(r'_\d+$', '', action_part)
    return normalize_action(action_part)


def action_to_cluster(action: str) -> Optional[str]:
    """Map normalized action → cluster, or None if unrecognized."""
    return _ACTION_TO_CLUSTER.get(normalize_action(action))


def is_other_label(action: str) -> bool:
    """True if the action is 'other' or unmappable to any cluster."""
    norm = normalize_action(action)
    return norm == 'other' or action_to_cluster(norm) is None


def disjoint_cluster_pool(target_cluster: str) -> list:
    """Return clusters that are semantically disjoint from target_cluster.
    Used for adversarial sampling."""
    if target_cluster == 'locomotion':
        return ['combat', 'idle', 'death', 'fly', 'jump', 'agitated', 'reaction']
    if target_cluster == 'combat':
        return ['locomotion', 'idle', 'death', 'fly', 'agitated', 'reaction']
    if target_cluster == 'idle':
        return ['locomotion', 'combat', 'fly', 'jump', 'agitated', 'swim']
    if target_cluster == 'death':
        return ['locomotion', 'combat', 'fly', 'jump', 'agitated']
    if target_cluster == 'fly':
        return ['locomotion', 'combat', 'idle', 'death', 'jump', 'swim']
    if target_cluster == 'takeoff':
        return ['locomotion', 'combat', 'idle', 'death', 'agitated']
    if target_cluster == 'jump':
        return ['locomotion', 'combat', 'idle', 'death', 'fly']
    if target_cluster == 'agitated':
        return ['locomotion', 'idle', 'fly', 'swim']
    if target_cluster == 'swim':
        return ['locomotion', 'combat', 'idle', 'fly', 'jump']
    if target_cluster == 'reaction':
        return ['locomotion', 'combat', 'fly', 'death']
    if target_cluster == 'other_kept':
        return ['combat', 'idle', 'death', 'fly', 'agitated']
    # Default: everything else
    return [c for c in ACTION_CLUSTERS if c != target_cluster]
