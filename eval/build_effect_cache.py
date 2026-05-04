"""Extract ψ for all Truebones clips + compute normalization + topology-gap split.

Outputs:
  eval/results/effect_cache/psi_all.npy            — [N_clips, 64, 62] effect tensors
  eval/results/effect_cache/clip_metadata.json     — fname, skeleton, motion_label per clip
  eval/results/effect_cache/psi_stats.json         — per-channel mean/std for z-scoring
  eval/results/effect_cache/topology_gap.npy       — [N_skel, N_skel] pairwise topology distance
  eval/results/effect_cache/split.json             — train/val split
"""
import os
import json
import numpy as np
import time
from os.path import join as pjoin


def topology_distance(parents_a, parents_b):
    """Distance between two skeletons. Combines joint count, depth, and degree."""
    Ja, Jb = len(parents_a), len(parents_b)
    joint_diff = abs(Ja - Jb) / max(Ja, Jb)

    def depth_dist(parents):
        J = len(parents)
        d = np.zeros(J)
        for j in range(J):
            depth = 0
            cur = j
            while parents[cur] != cur and parents[cur] >= 0 and depth < J:
                depth += 1
                cur = parents[cur]
            d[j] = depth
        return d

    da = depth_dist(parents_a)
    db = depth_dist(parents_b)
    max_depth_diff = abs(da.max() - db.max()) / max(da.max(), db.max(), 1)

    def degree_dist(parents):
        J = len(parents)
        deg = np.zeros(J)
        for j in range(J):
            p = parents[j]
            if 0 <= p < J and p != j:
                deg[p] += 1
        return deg

    deg_a, deg_b = degree_dist(parents_a), degree_dist(parents_b)
    max_deg_diff = abs(deg_a.max() - deg_b.max()) / max(deg_a.max(), deg_b.max(), 1)

    return float(joint_diff + max_depth_diff + max_deg_diff) / 3.0


def main():
    os.makedirs('eval/results/effect_cache', exist_ok=True)

    from data_loaders.truebones.truebones_utils.get_opt import get_opt
    from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np
    from eval.effect_program import extract_effect_program

    opt = get_opt(0)
    cond_dict = np.load(opt.cond_file, allow_pickle=True).item()
    with open('dataset/truebones/zoo/truebones_processed/motion_labels.json') as f:
        label_map = json.load(f)
    with open('dataset/truebones/zoo/truebones_processed/train_val_split.json') as f:
        existing_split = json.load(f)

    motion_files = sorted(f for f in os.listdir(opt.motion_dir) if f.endswith('.npy'))
    print(f"Found {len(motion_files)} motion files")

    all_psi = []
    metadata = []
    n_failed = 0
    n_skip = 0

    t0 = time.time()
    for i, fname in enumerate(motion_files):
        if fname not in label_map:
            n_skip += 1
            continue
        skel = label_map[fname]['skeleton']
        if skel not in cond_dict:
            n_skip += 1
            continue

        info = cond_dict[skel]
        n_joints = len(info['joints_names'])
        parents = info['parents'][:n_joints]
        offsets = info['offsets'][:n_joints]
        mean = info['mean'][:n_joints]
        std = info['std'][:n_joints]

        try:
            raw = np.load(pjoin(opt.motion_dir, fname))
            motion_denorm = raw[:, :n_joints] * (std + 1e-6) + mean
            positions = recover_from_bvh_ric_np(motion_denorm)

            eff = extract_effect_program(positions, parents, offsets)
            psi = eff['psi']

            if np.isnan(psi).any() or np.isinf(psi).any():
                n_failed += 1
                continue

            all_psi.append(psi)
            metadata.append({
                'fname': fname,
                'skeleton': skel,
                'coarse_label': label_map[fname].get('coarse_label', 'other'),
                'n_joints': n_joints,
                'split': 'train' if fname in set(existing_split['train']) else 'val',
            })

            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(motion_files) - i - 1) / rate
                print(f"  [{i+1}/{len(motion_files)}] failed={n_failed} skip={n_skip} "
                      f"rate={rate:.1f}/s eta={eta:.0f}s")
        except Exception as e:
            n_failed += 1
            if n_failed < 5:
                print(f"  fail {fname}: {e}")

    print(f"\nExtracted {len(all_psi)} ψ vectors ({n_failed} failed, {n_skip} skipped)")

    psi_arr = np.array(all_psi)
    print(f"ψ array shape: {psi_arr.shape}")

    np.save('eval/results/effect_cache/psi_all.npy', psi_arr)
    with open('eval/results/effect_cache/clip_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=1)

    # Per-channel statistics for z-scoring
    psi_flat = psi_arr.reshape(-1, 62)
    psi_mean = psi_flat.mean(axis=0)
    psi_std = psi_flat.std(axis=0) + 1e-6

    # Per-component breakdown
    component_slices = {
        'tau': (0, 8),
        'mu':  (8, 32),
        'eta': (32, 50),
        'rho': (50, 62),
    }
    comp_stats = {}
    for name, (lo, hi) in component_slices.items():
        sub = psi_flat[:, lo:hi]
        comp_stats[name] = {
            'mean': float(sub.mean()),
            'std':  float(sub.std()),
            'p1':   float(np.percentile(sub, 1)),
            'p99':  float(np.percentile(sub, 99)),
            'pct_zero': float(np.mean(np.abs(sub) < 1e-6)),
        }
    stats_out = {
        'mean':           psi_mean.tolist(),
        'std':            psi_std.tolist(),
        'component_stats': comp_stats,
        'n_clips':        len(all_psi),
    }
    with open('eval/results/effect_cache/psi_stats.json', 'w') as f:
        json.dump(stats_out, f, indent=2)

    # Topology gap matrix
    skels = sorted({m['skeleton'] for m in metadata})
    print(f"\nComputing topology gap for {len(skels)} skeletons...")
    gap = np.zeros((len(skels), len(skels)))
    for i, sa in enumerate(skels):
        pa = cond_dict[sa]['parents'][:len(cond_dict[sa]['joints_names'])]
        for j, sb in enumerate(skels):
            if i == j:
                continue
            pb = cond_dict[sb]['parents'][:len(cond_dict[sb]['joints_names'])]
            gap[i, j] = topology_distance(pa, pb)

    np.save('eval/results/effect_cache/topology_gap.npy', gap)
    with open('eval/results/effect_cache/topology_gap_skels.json', 'w') as f:
        json.dump(skels, f)

    print(f"Topology gap range: [{gap[gap > 0].min():.3f}, {gap.max():.3f}]")
    print(f"Topology gap mean:  {gap[gap > 0].mean():.3f}")

    # Topology-gap-aware split: hold out 5 skeleton families for OOD eval
    n_train_skels = int(0.7 * len(skels))
    rng = np.random.default_rng(42)
    skel_perm = rng.permutation(len(skels))
    train_skels = set(skels[i] for i in skel_perm[:n_train_skels])
    val_skels = set(skels[i] for i in skel_perm[n_train_skels:])

    # Per-clip split: train if (skeleton in train_skels) AND (clip in original train)
    n_train = sum(1 for m in metadata if m['skeleton'] in train_skels and m['split'] == 'train')
    n_val_indist = sum(1 for m in metadata if m['skeleton'] in train_skels and m['split'] == 'val')
    n_val_ood = sum(1 for m in metadata if m['skeleton'] in val_skels)

    split_out = {
        'train_skeletons':       sorted(train_skels),
        'val_skeletons':         sorted(val_skels),
        'n_train':               n_train,
        'n_val_indistribution':  n_val_indist,
        'n_val_ood_skeleton':    n_val_ood,
        'description': (
            "Three-way split: train on (train_skel × train_clip), "
            "val_indist on (train_skel × val_clip), "
            "val_ood on val_skel (entire skeleton held out)."
        ),
    }
    with open('eval/results/effect_cache/split.json', 'w') as f:
        json.dump(split_out, f, indent=2)

    print(f"\nSplit:")
    print(f"  Train: {len(train_skels)} skels × train clips = {n_train}")
    print(f"  Val (in-dist):   {len(train_skels)} skels × val clips = {n_val_indist}")
    print(f"  Val (OOD skel):  {len(val_skels)} skels × all clips = {n_val_ood}")

    print(f"\nDone. Saved to eval/results/effect_cache/")


if __name__ == '__main__':
    main()
