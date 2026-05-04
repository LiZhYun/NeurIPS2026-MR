"""Synthetic 2x2 dataset for causal isolation of gauge vs cell-mean mechanisms.

Builds 4 dataset cells crossing (supervision in {paired, unpaired}) x (cell density
in {sparse, dense}). Each dataset has a KNOWN ground-truth transport T*_{a->b} that
lets us compute the recovery error of any learned T-hat.

Skeleton family:
  - K=8 synthetic skeletons, chain topology, joint count J_s in {8, 10, 12, 14, 16,
    18, 20, 22}. Bone length 1.0 between consecutive joints. Root at origin.
  - Per-skeleton "rest pose" = chain along the y-axis.

Action vocabulary:
  - 6 actions, each = a parametric joint-angle trajectory over T=64 frames.
    locomotion: sinusoidal forward swing of all joints
    combat:     sharp swing of distal half
    idle:       small high-frequency wobble
    death:      monotonic collapse
    fly:        large-amplitude wave
    swim:       counter-phase wave on opposite halves

Ground-truth transport T*_{a->b}:
  - Per-frame, per-joint position resampling: x_b[t, j] = ResampleAlongChain(x_a[t, :], j/(J_b-1))
  - Plus a deterministic timing warp that depends only on action class (NOT on instance).
  - Therefore, T* is a function of (skel_a, skel_b, action_label) and NOT of the source
    instance: this is the "single-clip-per-bucket" setting under uniform pairing.

Density axis (cell |B(s, a)|):
  - sparse: 1 clip per (skeleton, action) cell. Total clips = K * 6 = 48.
  - dense:  50 clips per (skeleton, action) cell. Total clips = K * 6 * 50 = 2400.
    Within a cell, instances differ in random per-joint phase offsets and amplitude
    scales (i.e., the "intent" of the action is preserved but instance-level noise
    is added).

Supervision axis:
  - unpaired: each method sees only per-skeleton clip distributions {P_s}. No labels
    cross skeletons. (Action labels still available within a skeleton.)
  - paired:   each method sees (x_a, T*(x_a)) pairs sampled across all skel-pairs.

Output:
  save/synthetic_2x2/{cell_name}/
    clips.npy        # [N, T, J_max, 3]   padded to max joints, per-clip (J_s, action_id)
    meta.json        # {clips: [{skel: int, action: int, ...}], ...}
    transport.npy    # [N_pairs, T, J_max, 3] for paired cells; ground-truth T*(x_a)

Usage:
  python -m eval.synthetic_2x2.build_synth_dataset --out save/synthetic_2x2 --seed 0
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

K_SKELETONS = 8
JOINTS_PER_SKEL = [8, 10, 12, 14, 16, 18, 20, 22]
T_FRAMES = 64
ACTIONS = ['locomotion', 'combat', 'idle', 'death', 'fly', 'swim']


def make_chain_rest_pose(J: int) -> np.ndarray:
    """[J, 3] chain along y-axis, bone length 1.0."""
    p = np.zeros((J, 3), dtype=np.float32)
    p[:, 1] = np.arange(J, dtype=np.float32)
    return p


def action_trajectory(action: str, J: int, T: int, instance_seed: int,
                      add_instance_noise: bool) -> np.ndarray:
    """Generate one clip: [T, J, 3] joint positions for one action on a J-joint chain.

    The TIMING and SHAPE of the action are determined by the action label.
    Per-instance variation: phase + amplitude jitter, gated by add_instance_noise.
    """
    rng = np.random.RandomState(instance_seed)
    rest = make_chain_rest_pose(J)             # [J, 3]
    pose = np.broadcast_to(rest, (T, J, 3)).astype(np.float32).copy()
    t = np.arange(T, dtype=np.float32) / T     # [T] in [0, 1)

    phase = rng.uniform(0, 2*np.pi) if add_instance_noise else 0.0
    amp = (1.0 + rng.uniform(-0.1, 0.1)) if add_instance_noise else 1.0

    if action == 'locomotion':
        # Forward swing along x-axis for all joints, sinusoidal in time
        delta_x = amp * 0.5 * np.sin(2*np.pi*t + phase)              # [T]
        pose[..., 0] += delta_x[:, None]                              # broadcast over joints
    elif action == 'combat':
        # Sharp swing of distal half: square-ish wave on x for joints J/2..J-1
        delta_x = amp * np.sign(np.sin(4*np.pi*t + phase))            # [T]
        pose[:, J//2:, 0] += 0.6 * delta_x[:, None]
    elif action == 'idle':
        # Small high-frequency wobble on all joints, z-axis
        delta_z = amp * 0.05 * np.sin(8*np.pi*t + phase)              # [T]
        pose[..., 2] += delta_z[:, None]
    elif action == 'death':
        # Monotonic collapse: y-coordinate of distal joints drops from rest height
        decay = amp * (1.0 - t)                                       # [T] from 1 to 0
        for j in range(J):
            pose[:, j, 1] = j * decay                                 # gradual drop
    elif action == 'fly':
        # Large-amplitude wave along the chain, both x and z
        for j in range(J):
            pose[:, j, 0] += amp * 0.4 * np.sin(2*np.pi*t + phase + j*0.3)
            pose[:, j, 2] += amp * 0.4 * np.cos(2*np.pi*t + phase + j*0.3)
    elif action == 'swim':
        # Counter-phase wave: distal joints move opposite to proximal joints
        for j in range(J):
            sign = 1.0 if j < J//2 else -1.0
            pose[:, j, 0] += amp * sign * 0.3 * np.sin(2*np.pi*t + phase)
    else:
        raise ValueError(f"Unknown action: {action}")
    return pose


def true_transport(x_a: np.ndarray, J_b: int, action: str) -> np.ndarray:
    """Ground-truth T*_{a->b}(x_a): resample chain to J_b joints + apply timing warp.

    x_a: [T, J_a, 3]
    Returns: [T_b, J_b, 3]

    Resampling: linear interpolation along the chain index for each frame.
    Timing warp: action-specific affine reparametrization of t. (Same for all source
    instances of the same action, so this is the uniform-pairing assumption.)
    """
    T, J_a, _ = x_a.shape
    # Per-action timing warp: u(t) is a monotone function in [0, 1].
    t_grid = np.arange(T, dtype=np.float32) / max(1, T-1)                # source time in [0, 1]
    if action == 'locomotion':   u = t_grid                              # identity
    elif action == 'combat':     u = t_grid                              # identity
    elif action == 'idle':       u = t_grid                              # identity
    elif action == 'death':      u = np.power(t_grid, 1.5)               # ease-in
    elif action == 'fly':        u = t_grid                              # identity
    elif action == 'swim':       u = 0.5*(1 - np.cos(np.pi*t_grid))      # ease-in-out
    else: raise ValueError(action)
    # Map u back to source-frame index for indexing
    src_idx = np.clip((u * (T-1)).astype(np.int64), 0, T-1)
    x_a_warped = x_a[src_idx]                                            # [T, J_a, 3]

    # Resample chain to J_b joints by linear interpolation
    src_chain = np.linspace(0, 1, J_a, dtype=np.float32)
    tgt_chain = np.linspace(0, 1, J_b, dtype=np.float32)
    x_b = np.zeros((T, J_b, 3), dtype=np.float32)
    for d in range(3):
        for ti in range(T):
            x_b[ti, :, d] = np.interp(tgt_chain, src_chain, x_a_warped[ti, :, d])
    return x_b


def build_cell(out_dir: Path, density: str, supervision: str, seed: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    n_per_cell = 1 if density == 'sparse' else 50

    clips = []
    meta = []
    instance_id = 0
    for s_idx, J_s in enumerate(JOINTS_PER_SKEL):
        for a_idx, action in enumerate(ACTIONS):
            for k in range(n_per_cell):
                cs = seed * 1000003 + s_idx * 1009 + a_idx * 53 + k * 7
                clip = action_trajectory(action, J_s, T_FRAMES, cs,
                                         add_instance_noise=(n_per_cell > 1))
                clips.append(clip)
                meta.append({'instance_id': instance_id, 'skel': s_idx,
                             'action': a_idx, 'action_name': action,
                             'n_joints': J_s, 'cell_count': n_per_cell})
                instance_id += 1

    # Pad to common max-joint shape for stacking
    J_max = max(JOINTS_PER_SKEL)
    arr = np.zeros((len(clips), T_FRAMES, J_max, 3), dtype=np.float32)
    for i, clip in enumerate(clips):
        arr[i, :, :clip.shape[1], :] = clip
    np.save(out_dir / 'clips.npy', arr)

    # Build paired transport if supervision is paired
    if supervision == 'paired':
        # Sample N_pairs cross-skel-pairs, compute T*(x_a) per pair
        # For evenness, take min(N, K*K-K) pairs, ensuring all skeleton-pairs covered uniformly
        rng = np.random.RandomState(seed + 42)
        pair_records = []
        pair_arr_list = []
        for src_idx, item in enumerate(meta):
            sa, ai = item['skel'], item['action']
            # For each source clip, pair with 1 or 50 (matches density) target clips
            # of the same action on a different skeleton
            n_pairs_per_src = 1 if density == 'sparse' else min(5, K_SKELETONS - 1)
            for _ in range(n_pairs_per_src):
                sb = rng.choice([k for k in range(K_SKELETONS) if k != sa])
                xa = clips[src_idx]
                xb_target = true_transport(xa, JOINTS_PER_SKEL[sb], ACTIONS[ai])
                pair_records.append({
                    'src_instance_id': src_idx, 'skel_a': sa,
                    'skel_b': int(sb), 'action': ai,
                })
                pad = np.zeros((T_FRAMES, J_max, 3), dtype=np.float32)
                pad[:, :xb_target.shape[1], :] = xb_target
                pair_arr_list.append(pad)
        pair_arr = np.stack(pair_arr_list)
        np.save(out_dir / 'transport.npy', pair_arr)
        with open(out_dir / 'pairs.json', 'w') as f:
            json.dump(pair_records, f, indent=2)
        print(f"  paired transport: {pair_arr.shape}, {len(pair_records)} pair records")

    with open(out_dir / 'meta.json', 'w') as f:
        json.dump({
            'density': density, 'supervision': supervision,
            'seed': seed, 'n_clips': len(clips),
            'n_skeletons': K_SKELETONS, 'joints_per_skel': JOINTS_PER_SKEL,
            'n_actions': len(ACTIONS), 'actions': ACTIONS, 'T_frames': T_FRAMES,
            'J_max': J_max, 'clips': meta,
        }, f, indent=2)

    print(f"[build_cell] {density}_{supervision}: clips={arr.shape}, n={len(clips)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='save/synthetic_2x2')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    out_root = PROJECT_ROOT / args.out
    out_root.mkdir(parents=True, exist_ok=True)

    cells = [
        ('sparse', 'unpaired'),
        ('sparse', 'paired'),
        ('dense', 'unpaired'),
        ('dense', 'paired'),
    ]
    for density, supervision in cells:
        cell_dir = out_root / f'{density}_{supervision}'
        build_cell(cell_dir, density, supervision, args.seed)

    print(f"\nAll 4 cells written to {out_root}")


if __name__ == '__main__':
    main()
