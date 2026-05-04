"""Enhanced motion visualization with joint dots + per-limb bone coloring.

Renders motion to MP4 with:
  - Colored joint spheres (size proportional to joint importance)
  - Per-kinematic-chain bone coloring (different color per limb)
  - Ground plane
  - Title with skeleton name + metadata

Does NOT modify existing plot_script.py.

Usage:
    # Visualize a generated .npy file
    conda run -n anytop python -m eval.visualize_motion \
        --npy eval/results/generation_quality/official_samples/Horse_rep_0_#0.npy \
        --skeleton Horse

    # Batch: visualize all skeletons from official samples
    conda run -n anytop python -m eval.visualize_motion --batch_dir eval/results/generation_quality/official_samples
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors


LIMB_COLORS = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
    '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
    '#dcbeff', '#9A6324', '#800000', '#aaffc3', '#808000',
    '#000075', '#a9a9a9',
]

JOINT_COLOR = '#222222'
JOINT_SIZE_ROOT = 80
JOINT_SIZE_NORMAL = 30


def assign_chain_colors(parents, n_joints):
    """Assign a color to each joint based on its kinematic chain."""
    colors = ['#cccccc'] * n_joints
    chain_id = [0] * n_joints

    # Find chain roots: joints whose parent is 0 (root) or are root themselves
    current_chain = 0
    visited = [False] * n_joints
    children = [[] for _ in range(n_joints)]
    for j in range(n_joints):
        p = parents[j]
        if p >= 0 and p < n_joints and p != j:
            children[p].append(j)

    # BFS from root, assign chain IDs at branching points
    from collections import deque
    queue = deque([0])
    visited[0] = True
    chain_id[0] = 0

    while queue:
        node = queue.popleft()
        kids = children[node]
        if len(kids) > 1:
            # Branching point — each child starts a new chain
            for kid in kids:
                if not visited[kid]:
                    current_chain += 1
                    chain_id[kid] = current_chain
                    visited[kid] = True
                    queue.append(kid)
        else:
            for kid in kids:
                if not visited[kid]:
                    chain_id[kid] = chain_id[node]
                    visited[kid] = True
                    queue.append(kid)

    # Propagate chain IDs down remaining unvisited joints
    queue = deque([j for j in range(n_joints) if visited[j]])
    while queue:
        node = queue.popleft()
        for kid in children[node]:
            if not visited[kid]:
                chain_id[kid] = chain_id[node]
                visited[kid] = True
                queue.append(kid)

    n_chains = max(chain_id) + 1
    for j in range(n_joints):
        colors[j] = LIMB_COLORS[chain_id[j] % len(LIMB_COLORS)]
    return colors, chain_id


def render_motion_mp4(positions, parents, n_joints, save_path,
                      title='', fps=30, figsize=(8, 8)):
    """Render motion positions to MP4 with joint dots + colored bones.

    positions: [T, J, 3] global joint positions (denormalized)
    parents: [J] parent indices
    n_joints: number of real joints (rest are padding)
    """
    pos = positions[:, :n_joints, :]  # [T, J_real, 3]
    T = pos.shape[0]

    bone_colors, _ = assign_chain_colors(parents[:n_joints], n_joints)

    # Compute bounding box for camera
    all_pos = pos.reshape(-1, 3)
    center = all_pos.mean(axis=0)
    extent = max(all_pos.ptp(axis=0)) * 0.6 + 0.1

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):
        ax.clear()
        ax.set_xlim3d(center[0] - extent, center[0] + extent)
        ax.set_ylim3d(center[1] - extent, center[1] + extent)
        ax.set_zlim3d(center[2] - extent, center[2] + extent)
        ax.view_init(elev=20, azim=frame * 0.5 - 90)  # slow rotation
        ax.set_title(f'{title}  frame {frame}/{T}', fontsize=10)
        ax.set_axis_off()

        p = pos[frame]  # [J, 3]

        # Draw ground plane
        xx, zz = np.meshgrid(
            np.linspace(center[0] - extent, center[0] + extent, 2),
            np.linspace(center[2] - extent, center[2] + extent, 2))
        yy = np.full_like(xx, pos[:, :, 1].min() - 0.02)
        ax.plot_surface(xx, yy, zz, alpha=0.1, color='gray')

        # Draw bones
        par = parents[:n_joints]
        for j in range(1, n_joints):
            parent = par[j]
            if parent < 0 or parent >= n_joints or parent == j:
                continue
            ax.plot3D(
                [p[j, 0], p[parent, 0]],
                [p[j, 1], p[parent, 1]],
                [p[j, 2], p[parent, 2]],
                color=bone_colors[j], linewidth=2.0, alpha=0.8)

        # Draw joints
        for j in range(n_joints):
            size = JOINT_SIZE_ROOT if j == 0 else JOINT_SIZE_NORMAL
            ax.scatter(p[j, 0], p[j, 1], p[j, 2],
                       c=bone_colors[j], s=size, marker='o',
                       edgecolors='black', linewidths=0.3, alpha=0.9,
                       depthshade=True)

    anim = FuncAnimation(fig, update, frames=T, interval=1000/fps, blit=False)
    anim.save(save_path, writer='ffmpeg', fps=fps, dpi=100)
    plt.close(fig)
    print(f"  Saved → {save_path}")


def visualize_npy(npy_path, skeleton_name, cond_dict, out_dir, fps=30):
    """Load a generated .npy motion, recover positions, and render."""
    from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np

    motion = np.load(npy_path)  # [T, J, 13] (denormalized by generate.py)
    T, J, _ = motion.shape

    info = cond_dict[skeleton_name]
    parents = info['parents']
    n_joints = len(info['joints_names'])

    # Recover global joint positions from the motion representation
    positions = recover_from_bvh_ric_np(motion[:, :n_joints])  # [T, J, 3]

    basename = os.path.splitext(os.path.basename(npy_path))[0]
    save_path = os.path.join(out_dir, f'{basename}_enhanced.mp4')
    render_motion_mp4(positions, parents, n_joints, save_path,
                      title=f'{skeleton_name} ({n_joints}j)', fps=fps)


def visualize_real(skeleton_name, cond_dict, motion_dir, out_dir, n_frames=80, fps=30):
    """Visualize one real motion clip for comparison."""
    from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np

    info = cond_dict[skeleton_name]
    n_joints = len(info['joints_names'])
    parents = info['parents']
    mean = info['mean']
    std = info['std']

    motion_files = sorted(f for f in os.listdir(motion_dir)
                          if f.startswith(f'{skeleton_name}_'))
    if not motion_files:
        print(f"  No real motions for {skeleton_name}")
        return

    raw = np.load(os.path.join(motion_dir, motion_files[0]))  # [T, J, 13] normalized
    # Denormalize
    motion = raw[:n_frames, :n_joints] * (std[:n_joints] + 1e-6) + mean[:n_joints]
    positions = recover_from_bvh_ric_np(motion)

    save_path = os.path.join(out_dir, f'{skeleton_name}_real_enhanced.mp4')
    render_motion_mp4(positions, parents, n_joints, save_path,
                      title=f'{skeleton_name} REAL ({n_joints}j)', fps=fps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npy', type=str, default='', help='Single .npy to visualize')
    parser.add_argument('--skeleton', type=str, default='', help='Skeleton name for --npy')
    parser.add_argument('--batch_dir', type=str, default='', help='Batch: dir of generated .npy files')
    parser.add_argument('--include_real', action='store_true', help='Also render one real clip per skeleton')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    from data_loaders.truebones.truebones_utils.get_opt import get_opt
    opt = get_opt(args.device)
    cond_dict = np.load(opt.cond_file, allow_pickle=True).item()

    if args.npy:
        out_dir = os.path.dirname(args.npy) or '.'
        visualize_npy(args.npy, args.skeleton, cond_dict, out_dir, args.fps)
        return

    if args.batch_dir:
        out_dir = os.path.join(args.batch_dir, 'enhanced')
        os.makedirs(out_dir, exist_ok=True)

        npy_files = sorted(f for f in os.listdir(args.batch_dir) if f.endswith('.npy'))
        seen_skels = set()
        for npy_file in npy_files:
            skel = npy_file.split('_rep_')[0]
            if skel not in cond_dict:
                print(f"  Skipping {npy_file} — skeleton '{skel}' not in cond_dict")
                continue
            seen_skels.add(skel)
            print(f"Rendering {npy_file}...")
            visualize_npy(os.path.join(args.batch_dir, npy_file),
                          skel, cond_dict, out_dir, args.fps)

        if args.include_real:
            for skel in sorted(seen_skels):
                print(f"Rendering real {skel}...")
                visualize_real(skel, cond_dict, opt.motion_dir, out_dir, fps=args.fps)

        print(f"\nAll rendered to {out_dir}/")


if __name__ == '__main__':
    main()
