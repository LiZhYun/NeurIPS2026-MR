"""K_retrieve ablations: which Q components drive retrieval quality?

Variants:
  - full:      current q_signature (all 5 components)
  - com_only:  only COM path features
  - contact_only: only contact schedule features
  - cadence_only: only cadence
  - limb_only: only limb usage
  - no_com:    everything except COM
  - no_contact: everything except contact
  - oracle_action: retrieve any clip with same action label (upper bound)

Outputs per variant: eval/results/k_compare/K_retrieve_<variant>_200pair/pair_NNNN.npy
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MANIFEST = PROJECT_ROOT / 'eval/benchmark_paired/pairs/manifest.json'

# Q signature layout: [com_sum(6), hv_sum(4), cs_sum(4), cadence(1), limb(5)] = 20 dims
# Actually re-check — let's compute dimensions
SIG_COM = slice(0, 6)
SIG_HV = slice(6, 10)
SIG_CS = slice(10, 14)
SIG_CAD = slice(14, 15)
SIG_LIMB = slice(15, 20)


def mask_signature(sig: np.ndarray, variant: str) -> np.ndarray:
    """Zero out components not used by this variant."""
    s = sig.copy()
    if variant == 'full':
        return s
    if variant == 'com_only':
        mask = np.zeros_like(s)
        mask[SIG_COM] = 1.0
        return s * mask
    if variant == 'contact_only':
        mask = np.zeros_like(s)
        mask[SIG_CS] = 1.0
        return s * mask
    if variant == 'cadence_only':
        mask = np.zeros_like(s)
        mask[SIG_CAD] = 1.0
        return s * mask
    if variant == 'limb_only':
        mask = np.zeros_like(s)
        mask[SIG_LIMB] = 1.0
        return s * mask
    if variant == 'no_com':
        s[SIG_COM] = 0.0
        return s
    if variant == 'no_contact':
        s[SIG_CS] = 0.0
        return s
    raise ValueError(f'unknown variant: {variant}')


def parse_action_coarse(fname: str) -> str:
    """Extract coarse action label from filename, stripping trailing digit variants.

    'Skel___Attack2_1039.npy' → 'attack'
    'Skel___AttackRight_921.npy' → 'attackright'
    'Skel___Walk_11.npy' → 'walk'
    """
    import re
    parts = fname.split('___')
    if len(parts) >= 2:
        raw = parts[1].rsplit('_', 1)[0].lower()
        # Strip trailing digit variants (attack1, attack2 → attack)
        stripped = re.sub(r'\d+$', '', raw)
        return stripped if stripped else raw
    return fname.lower()


def run(variant: str, max_pairs=200, seed=42):
    out_dir = PROJECT_ROOT / f'eval/results/k_compare/K_retrieve_{variant}_200pair'
    out_dir.mkdir(parents=True, exist_ok=True)

    from eval.run_k_pipeline_200pairs import load_assets
    from eval.quotient_extractor import extract_quotient
    from eval.pilot_Q_experiments import q_signature
    from eval.run_k_pipeline_30pairs import build_q_star
    from eval.run_k_retrieve_refine_200pairs import list_target_skel_motions

    cond, contact_groups, motion_dir = load_assets()

    Q_CACHE_PATH = PROJECT_ROOT / 'idea-stage/quotient_cache.npz'
    qc = np.load(Q_CACHE_PATH, allow_pickle=True)
    qmeta = list(qc['meta'])
    qfname_to_idx = {m['fname']: i for i, m in enumerate(qmeta)}

    def get_q_sig(fname):
        idx = qfname_to_idx.get(fname)
        if idx is None:
            return None
        q = {
            'com_path': qc['com_path'][idx],
            'heading_vel': qc['heading_vel'][idx],
            'contact_sched': qc['contact_sched'][idx],
            'cadence': float(qc['cadence'][idx]),
            'limb_usage': qc['limb_usage'][idx],
        }
        return q_signature(q)

    with open(MANIFEST) as f:
        manifest = json.load(f)
    pairs = manifest['pairs'][:max_pairs]
    print(f"[{variant}] {len(pairs)} pairs")

    tgt_clip_cache = {}
    rng = np.random.RandomState(seed)

    per_pair = []
    t0_total = time.time()

    for i, p in enumerate(pairs):
        pid = p['pair_id']
        src_skel = p['skel_a']
        tgt_skel = p['skel_b']
        src_fname = p['file_a']
        tgt_fname = p['file_b']
        action = p['action']

        rec = {
            'pair_id': pid, 'action': action,
            'source_skel': src_skel, 'target_skel': tgt_skel,
            'status': 'pending',
        }

        if src_skel not in contact_groups or tgt_skel not in contact_groups:
            rec['status'] = 'skipped_no_cg'
            per_pair.append(rec)
            continue

        try:
            # Build Q* signature
            q_src = extract_quotient(src_fname, cond[src_skel],
                                     contact_groups=contact_groups,
                                     motion_dir=motion_dir)
            q_star = build_q_star(q_src, src_skel, tgt_skel, contact_groups, cond)
            q_star_sig_full = q_signature(q_star)

            if tgt_skel not in tgt_clip_cache:
                tgt_clip_cache[tgt_skel] = list_target_skel_motions(tgt_skel, motion_dir)
            full_pool = tgt_clip_cache[tgt_skel]
            if tgt_fname not in full_pool:
                raise RuntimeError(f'GT not in pool for {tgt_skel}')
            tgt_files = [f for f in full_pool if f != tgt_fname]
            if not tgt_files:
                raise RuntimeError('empty pool after GT exclusion')

            if variant == 'oracle_action':
                # Retrieve clips with same coarse action label (Attack1/Attack2 → attack)
                action_matches = [f for f in tgt_files if parse_action_coarse(f) == action]
                if action_matches:
                    best_fname = action_matches[rng.randint(0, len(action_matches))]
                    rec['oracle_label_match'] = True
                    rec['n_action_matches'] = len(action_matches)
                else:
                    rec['status'] = 'skipped_no_action_match'
                    per_pair.append(rec)
                    continue
                rec['retrieval_cosine'] = None  # not applicable for oracle
            else:
                # Q-component variants: mask signature and find best
                masked_q_star = mask_signature(q_star_sig_full, variant)
                q_star_norm = masked_q_star / (np.linalg.norm(masked_q_star) + 1e-9)
                best_sim = -np.inf
                best_fname = None
                for cand_fname in tgt_files:
                    cand_sig = get_q_sig(cand_fname)
                    if cand_sig is None:
                        continue
                    masked = mask_signature(cand_sig, variant)
                    masked_norm = masked / (np.linalg.norm(masked) + 1e-9)
                    sim = float(q_star_norm @ masked_norm)
                    if sim > best_sim:
                        best_sim = sim
                        best_fname = cand_fname
                if best_fname is None:
                    raise RuntimeError('no valid candidate')
                rec['retrieval_cosine'] = best_sim

            assert best_fname != tgt_fname
            rec['retrieved_fname'] = best_fname

            # Save retrieved motion directly (no refinement)
            motion = np.load(os.path.join(motion_dir, best_fname)).astype(np.float32)
            out_path = out_dir / f'pair_{pid:04d}.npy'
            np.save(out_path, motion)
            rec['status'] = 'ok'

            if (i + 1) % 20 == 0 or i == 0:
                elapsed = time.time() - t0_total
                eta = elapsed / (i + 1) * (len(pairs) - i - 1)
                print(f"  [{i+1}/{len(pairs)}] {action} sim={rec['retrieval_cosine']:.3f} "
                      f"(elapsed {elapsed:.0f}s, ETA {eta:.0f}s)")

        except Exception as e:
            rec['status'] = 'failed'
            rec['error'] = str(e)
            print(f"  FAILED pair {pid}: {e}")

        per_pair.append(rec)

    total_time = time.time() - t0_total
    print(f"[{variant}] Total: {total_time:.0f}s")

    summary = {
        'method': f'K_retrieve_{variant}',
        'variant': variant,
        'n_pairs': len(per_pair),
        'n_ok': sum(1 for r in per_pair if r['status'] == 'ok'),
        'total_time_sec': total_time,
        'per_pair': per_pair,
    }
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, required=True,
                        choices=['full', 'com_only', 'contact_only', 'cadence_only',
                                 'limb_only', 'no_com', 'no_contact',
                                 'oracle_action'])
    parser.add_argument('--max_pairs', type=int, default=200)
    args = parser.parse_args()
    run(variant=args.variant, max_pairs=args.max_pairs)
