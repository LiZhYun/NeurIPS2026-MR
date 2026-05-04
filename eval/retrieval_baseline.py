"""Retrieval baseline (per Round 7 review — CRITICAL anti-trivial baseline).

For each (source, target_skel) pair, retrieve the nearest real target-skel clip by B-similarity.
If our trained model doesn't beat this, we've only learned similarity search, not generation.

B-similarity = cosine similarity in the (a, ψ)-space (action label + analytic dynamics).

Usage:
    conda run -n anytop python -m eval.retrieval_baseline \
        --pairs eval/results/track_b/generated_motions.npz \
        --out eval/results/track_b/retrieval_baseline.npz
"""
import os
import json
import argparse
import numpy as np
from os.path import join as pjoin


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pairs', required=True, help='npz from track_b_inference (uses pairs metadata only)')
    p.add_argument('--effect_cache', default='eval/results/effect_cache/psi_all.npy')
    p.add_argument('--clip_metadata', default='eval/results/effect_cache/clip_metadata.json')
    p.add_argument('--out', default='eval/results/track_b/retrieval_baseline.npz')
    return p.parse_args()


ACTION_CLASSES = ['walk', 'run', 'idle', 'attack', 'fly', 'swim', 'jump',
                  'turn', 'die', 'eat', 'getup', 'other']
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_CLASSES)}


def main():
    args = parse_args()
    from data_loaders.truebones.truebones_utils.get_opt import get_opt
    from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np

    print(f"Loading {args.pairs}")
    data = np.load(args.pairs, allow_pickle=True)
    pairs = data['pairs']

    # Load ψ cache and metadata
    psi_all = np.load(args.effect_cache)
    with open(args.clip_metadata) as f:
        metadata = json.load(f)
    fname_to_idx = {m['fname']: i for i, m in enumerate(metadata)}
    fname_to_action = {m['fname']: ACTION_TO_IDX.get(m['coarse_label'], ACTION_TO_IDX['other'])
                       for m in metadata}

    # Index target-skel clips by (skeleton, fname)
    target_skel_to_clips = {}
    for i, m in enumerate(metadata):
        target_skel_to_clips.setdefault(m['skeleton'], []).append(i)

    opt = get_opt(0)
    cond_dict = np.load(opt.cond_file, allow_pickle=True).item()

    # For each pair: source_psi + source_action, find nearest target-skel clip
    print(f"Retrieving for {len(pairs)} pairs...")
    motions = []
    valid_pairs = []
    for pair in pairs:
        src_fname, src_skel, tgt_skel = pair[0], pair[1], pair[2]
        if src_fname not in fname_to_idx:
            continue
        if tgt_skel not in target_skel_to_clips:
            continue
        if tgt_skel not in cond_dict:
            continue

        src_psi = psi_all[fname_to_idx[src_fname]]
        src_action = fname_to_action[src_fname]

        # Find nearest target-skel clip by combined ψ similarity + action match preference
        candidate_indices = target_skel_to_clips[tgt_skel]
        # Skip same-fname (shouldn't happen since src_skel != tgt_skel) but defensive
        candidate_indices = [i for i in candidate_indices if metadata[i]['fname'] != src_fname]
        if not candidate_indices:
            continue

        # Cosine similarity between source ψ and each candidate's ψ
        cand_psi = psi_all[candidate_indices]  # [N, 64, 62]
        # Flatten + L2 normalize
        src_flat = src_psi.flatten()
        src_norm = src_flat / (np.linalg.norm(src_flat) + 1e-8)
        cand_flat = cand_psi.reshape(len(candidate_indices), -1)
        cand_norm = cand_flat / (np.linalg.norm(cand_flat, axis=1, keepdims=True) + 1e-8)
        sims = cand_norm @ src_norm

        # Boost candidates with matching action
        cand_actions = np.array([fname_to_action[metadata[i]['fname']] for i in candidate_indices])
        action_bonus = (cand_actions == src_action).astype(np.float32) * 0.3
        sims = sims + action_bonus

        best_idx = candidate_indices[int(np.argmax(sims))]
        best_fname = metadata[best_idx]['fname']

        # Load the actual motion (denormalized positions for evaluator)
        info = cond_dict[tgt_skel]
        n_joints = len(info['joints_names'])
        mean = info['mean'][:n_joints]
        std = info['std'][:n_joints] + 1e-6
        try:
            raw = np.load(pjoin(opt.motion_dir, best_fname))
            # denormalized 13-dim features [T, J, 13]
            motion_denorm = raw[:, :n_joints] * std + mean
            motions.append(motion_denorm)
            valid_pairs.append((src_fname, src_skel, tgt_skel))
        except Exception as e:
            continue

    print(f"  Retrieved {len(motions)} valid motions")
    # Save as object array (heterogeneous shapes)
    motions_obj = np.empty(len(motions), dtype=object)
    for i, m in enumerate(motions):
        motions_obj[i] = m.astype(np.float32)
    pairs_obj = np.array(valid_pairs, dtype=object)
    np.savez(args.out, pairs=pairs_obj, motions=motions_obj, allow_pickle=True)
    print(f"Saved → {args.out}")


if __name__ == '__main__':
    main()
