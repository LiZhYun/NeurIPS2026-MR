"""Topology editing pipeline for Track A synthetic controlled benchmark.

Generates pseudo-paired (x_S, x_S') data via local topology edits with KNOWN target motion.
Each edit operation is deterministic so the target motion is unambiguous.

Edit operations (all preserve motion semantics, so synthetic ground truth is well-defined):
  1. SUBDIVIDE_BONE: insert a midpoint joint along a bone, motion at new joint = midpoint of parent/child
  2. MERGE_REDUNDANT: remove a degree-2 intermediate joint, blend its motion into neighbors
  3. PRUNE_LEAF: remove a leaf joint, motion preserved for all remaining joints
  4. DUPLICATE_LEAF: clone a leaf joint with same motion (e.g., add a co-located finger)

These are TRIVIAL edits where DEFORM is well-defined. They test local parametric robustness,
NOT true cross-morphology transfer (per topology_editing_deep_analysis.md). Used for Track A
sanity check, not as primary claim.

Usage:
    conda run -n anytop python -m eval.topology_editing --build
"""
import os
import json
import argparse
import numpy as np
from os.path import join as pjoin


def subdivide_bone(parents, offsets, motion_positions, child_idx):
    """Insert a midpoint joint between child_idx and its parent.

    Returns: new parents, new offsets, new motion (with midpoint motion = avg of parent/child)
    """
    parent_idx = parents[child_idx]
    if parent_idx < 0 or parent_idx == child_idx:
        return None  # cannot subdivide root or self-loop

    J = len(parents)
    new_J = J + 1
    new_idx = J  # new midpoint joint at end

    new_parents = np.zeros(new_J, dtype=parents.dtype)
    new_parents[:J] = parents.copy()
    new_parents[child_idx] = new_idx  # child's parent is now the midpoint
    new_parents[new_idx] = parent_idx  # midpoint's parent is the original parent

    new_offsets = np.zeros((new_J, 3), dtype=offsets.dtype)
    new_offsets[:J] = offsets.copy()
    new_offsets[new_idx] = offsets[child_idx] / 2.0  # midpoint at half offset
    new_offsets[child_idx] = offsets[child_idx] / 2.0  # child's offset shortened

    T = motion_positions.shape[0]
    new_motion = np.zeros((T, new_J, 3), dtype=motion_positions.dtype)
    new_motion[:, :J] = motion_positions
    new_motion[:, new_idx] = (motion_positions[:, parent_idx] + motion_positions[:, child_idx]) / 2.0

    return new_parents, new_offsets, new_motion


def prune_leaf(parents, offsets, motion_positions):
    """Remove a random leaf joint."""
    J = len(parents)
    children = [[] for _ in range(J)]
    for j in range(J):
        p = parents[j]
        if 0 <= p < J and p != j:
            children[p].append(j)
    leaves = [j for j in range(J) if not children[j] and j != 0]  # not root
    if not leaves:
        return None
    leaf = leaves[0]  # deterministic: prune first leaf

    new_parents = np.delete(parents, leaf)
    new_offsets = np.delete(offsets, leaf, axis=0)
    new_motion = np.delete(motion_positions, leaf, axis=1)
    # Remap parent indices that pointed beyond the removed leaf
    new_parents = np.where(new_parents > leaf, new_parents - 1, new_parents)
    return new_parents, new_offsets, new_motion


def duplicate_leaf(parents, offsets, motion_positions):
    """Duplicate a leaf joint: add a clone at the same offset and same motion."""
    J = len(parents)
    children = [[] for _ in range(J)]
    for j in range(J):
        p = parents[j]
        if 0 <= p < J and p != j:
            children[p].append(j)
    leaves = [j for j in range(J) if not children[j] and j != 0]
    if not leaves:
        return None
    leaf = leaves[0]

    new_J = J + 1
    new_parents = np.zeros(new_J, dtype=parents.dtype)
    new_parents[:J] = parents.copy()
    new_parents[J] = parents[leaf]  # duplicate has same parent as original leaf
    new_offsets = np.zeros((new_J, 3), dtype=offsets.dtype)
    new_offsets[:J] = offsets
    new_offsets[J] = offsets[leaf]  # same offset
    T = motion_positions.shape[0]
    new_motion = np.zeros((T, new_J, 3), dtype=motion_positions.dtype)
    new_motion[:, :J] = motion_positions
    new_motion[:, J] = motion_positions[:, leaf]  # same motion as original leaf
    return new_parents, new_offsets, new_motion


def merge_redundant(parents, offsets, motion_positions):
    """Remove a degree-2 intermediate joint (parent has 1 child, that child has 1 child)."""
    J = len(parents)
    children = [[] for _ in range(J)]
    for j in range(J):
        p = parents[j]
        if 0 <= p < J and p != j:
            children[p].append(j)
    # Find a degree-2 chain
    candidates = [j for j in range(1, J)
                  if len(children[j]) == 1 and parents[j] >= 0 and parents[j] != j]
    if not candidates:
        return None
    mid = candidates[0]
    parent = parents[mid]
    child = children[mid][0]

    # Skip mid: child's parent becomes parent. mid is removed.
    new_parents = np.delete(parents, mid).copy()
    new_offsets = np.delete(offsets, mid, axis=0).copy()
    new_motion = np.delete(motion_positions, mid, axis=1).copy()

    # Remap: any parent index > mid → -1, the child of mid becomes parent's child
    # First adjust the child indexing
    child_new_idx = child if child < mid else child - 1
    parent_new_idx = parent if parent < mid else parent - 1
    new_parents[child_new_idx] = parent_new_idx
    new_parents = np.where(new_parents > mid, new_parents - 1, new_parents)
    # Combine offsets at the merged location
    new_offsets[child_new_idx] = offsets[child] + offsets[mid]

    return new_parents, new_offsets, new_motion


EDIT_OPS = {
    'subdivide': lambda p, o, m: subdivide_bone(p, o, m, min(2, len(p)-1)),
    'prune': prune_leaf,
    'duplicate': duplicate_leaf,
    'merge': merge_redundant,
}


def build_synthetic_pairs(out_dir, n_pairs_per_op=50, seed=42):
    """For each operation × N source clips, build (source, edited) pairs with known motion."""
    from data_loaders.truebones.truebones_utils.get_opt import get_opt
    from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np
    rng = np.random.default_rng(seed)

    opt = get_opt(0)
    cond_dict = np.load(opt.cond_file, allow_pickle=True).item()
    with open('dataset/truebones/zoo/truebones_processed/motion_labels.json') as f:
        label_map = json.load(f)
    with open('dataset/truebones/zoo/truebones_processed/train_val_split.json') as f:
        split = json.load(f)

    val_files = split['val']
    pairs = []

    for op_name, op_fn in EDIT_OPS.items():
        n_kept = 0
        idx = 0
        while n_kept < n_pairs_per_op and idx < len(val_files):
            fname = val_files[idx]
            idx += 1
            if fname not in label_map:
                continue
            skel = label_map[fname]['skeleton']
            if skel not in cond_dict:
                continue
            info = cond_dict[skel]
            n_joints = len(info['joints_names'])
            if n_joints < 4:
                continue

            parents = np.array(info['parents'][:n_joints], dtype=np.int64)
            offsets = info['offsets'][:n_joints]
            mean = info['mean'][:n_joints]
            std = info['std'][:n_joints]

            try:
                raw = np.load(pjoin(opt.motion_dir, fname))
                motion_denorm = raw[:, :n_joints] * (std + 1e-6) + mean
                positions = recover_from_bvh_ric_np(motion_denorm)
            except Exception:
                continue

            result = op_fn(parents, offsets, positions)
            if result is None:
                continue
            new_parents, new_offsets, new_positions = result

            pairs.append({
                'op': op_name,
                'source_fname': fname,
                'source_skel': skel,
                'source_n_joints': int(n_joints),
                'target_n_joints': int(len(new_parents)),
                'motion_label': label_map[fname].get('coarse_label', 'other'),
            })

            # Save the edited (skeleton, motion) pair
            np.savez(
                pjoin(out_dir, f'{op_name}_{n_kept:04d}.npz'),
                source_positions=positions,
                source_parents=parents,
                source_offsets=offsets,
                target_positions=new_positions,
                target_parents=new_parents,
                target_offsets=new_offsets,
                source_fname=fname,
                source_skel=skel,
                op=op_name,
            )
            n_kept += 1
        print(f"  {op_name}: {n_kept} pairs built")

    with open(pjoin(out_dir, 'pairs_metadata.json'), 'w') as f:
        json.dump(pairs, f, indent=1)
    print(f"\nTotal {len(pairs)} synthetic pairs saved to {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default='eval/results/track_a_synthetic')
    p.add_argument('--n_pairs_per_op', type=int, default=50)
    p.add_argument('--build', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    if args.build:
        os.makedirs(args.out_dir, exist_ok=True)
        build_synthetic_pairs(args.out_dir, args.n_pairs_per_op, args.seed)
    else:
        print("Use --build to generate synthetic pairs")


if __name__ == '__main__':
    main()
