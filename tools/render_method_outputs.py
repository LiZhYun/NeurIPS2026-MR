"""Render motion outputs (13-dim per-joint tensors) to MP4 using matplotlib 3D.

For each pair in idea-stage/eval_pairs.json and each method in
{K, q_retrieval, q_label_retrieval, motion2motion, label_random}, render the
motion output at
  eval/results/k_compare/<method>/pair_<id>_<src>_to_<tgt>.npy
to
  eval/results/k_compare/renders/<method>/pair_<id>.mp4

Also renders each source motion to renders/source/pair_<id>.mp4.

Motion tensors are (T, J, 13); the first 3 dims are root-relative XYZ
positions. Skeleton parents come from cond.npy.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: F401

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COND_PATH = PROJECT_ROOT / "dataset" / "truebones" / "zoo" / "truebones_processed" / "cond.npy"
EVAL_PAIRS_PATH = PROJECT_ROOT / "idea-stage" / "eval_pairs.json"
K_COMPARE_DIR = PROJECT_ROOT / "eval" / "results" / "k_compare"
SOURCE_MOTION_DIR = PROJECT_ROOT / "dataset" / "truebones" / "zoo" / "truebones_processed" / "motions"

METHODS = ["K", "q_retrieval", "q_label_retrieval", "motion2motion", "label_random"]


def load_cond():
    return np.load(COND_PATH, allow_pickle=True).item()


def pick_radius(positions):
    """Pick a rendering radius from the motion bounding box."""
    span = max(
        positions[..., 0].max() - positions[..., 0].min(),
        positions[..., 1].max() - positions[..., 1].min(),
        positions[..., 2].max() - positions[..., 2].min(),
    )
    return max(1.0, float(span) * 1.1)


def render_motion_to_mp4(positions, parents, out_path, fps=30, figsize=(4, 4)):
    """Render (T, J, 3) joint positions as a line-segment 3D animation.

    Uses matplotlib's FFMpegWriter (h264). Ground plane drawn at min-Y.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pos = np.asarray(positions, dtype=np.float32).copy()
    T, J, _ = pos.shape
    assert len(parents) == J, f"parents {len(parents)} vs joints {J}"

    # Normalize: center ground on min-Y, flatten root XY trajectory to origin for each frame
    height_offset = pos[:, :, 1].min()
    pos[:, :, 1] -= height_offset
    # Keep root horizontal trajectory so we see locomotion
    radius = pick_radius(pos)

    mins = pos.min(axis=(0, 1))
    maxs = pos.max(axis=(0, 1))

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    def setup_axes():
        ax.set_xlim3d([mins[0] - 0.1, maxs[0] + 0.1])
        ax.set_ylim3d([0, max(maxs[1], 0.5) + 0.1])
        ax.set_zlim3d([mins[2] - 0.1, maxs[2] + 0.1])
        # matplotlib 3.1 lacks set_box_aspect; best-effort
        try:
            ax.set_box_aspect((1, 1, 1))  # type: ignore[attr-defined]
        except AttributeError:
            pass
        ax.view_init(elev=20, azim=-60)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")

    bone_pairs = [(i, int(p)) for i, p in enumerate(parents) if p >= 0]

    def update(frame):
        ax.clear()
        setup_axes()
        # Ground plane
        x0, x1 = mins[0] - 0.2, maxs[0] + 0.2
        z0, z1 = mins[2] - 0.2, maxs[2] + 0.2
        verts = [[[x0, 0, z0], [x1, 0, z0], [x1, 0, z1], [x0, 0, z1]]]
        plane = Poly3DCollection(verts, alpha=0.15, facecolor=(0.5, 0.5, 0.5))
        ax.add_collection3d(plane)
        # Bones
        p = pos[frame]
        for a, b in bone_pairs:
            ax.plot3D(
                [p[a, 0], p[b, 0]],
                [p[a, 1], p[b, 1]],
                [p[a, 2], p[b, 2]],
                color="#DD5A37",
                linewidth=2.0,
                solid_capstyle="round",
            )
        # Joint dots
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], c="#222222", s=6)
        ax.set_title(f"frame {frame + 1}/{T}", fontsize=8)
        return []

    ani = animation.FuncAnimation(fig, update, frames=T, interval=1000 / fps, blit=False)
    writer = animation.FFMpegWriter(fps=fps, codec="libx264", bitrate=1200)
    ani.save(str(out_path), writer=writer, dpi=90)
    plt.close(fig)


def get_source_positions(source_fname, cond):
    """Load source motion, return (T, J, 3) joint positions."""
    src_path = SOURCE_MOTION_DIR / source_fname
    if not src_path.exists():
        raise FileNotFoundError(f"Source motion not found: {src_path}")
    motion = np.load(src_path, allow_pickle=True)
    if motion.dtype == object:
        motion = motion.item()
        if isinstance(motion, dict):
            motion = motion.get("motion", motion)
    # Should be (T, J, 13) or (J, 13, T)
    if motion.ndim == 3 and motion.shape[-1] == 13:
        return motion[..., :3]
    if motion.ndim == 3 and motion.shape[1] == 13:
        # (J, 13, T) -> (T, J, 3)
        return np.transpose(motion[:, :3, :], (2, 0, 1))
    raise ValueError(f"Unexpected source motion shape: {motion.shape}")


def get_method_positions(method, pair_id, src, tgt):
    """Load method output (T, J_tgt, 13) and return positions."""
    # npy files named pair_<id>_<src>_to_<tgt>.npy (id is two-digit)
    fname = f"pair_{pair_id:02d}_{src}_to_{tgt}.npy"
    fpath = K_COMPARE_DIR / method / fname
    if not fpath.exists():
        # Scan for any file with matching prefix
        matches = sorted((K_COMPARE_DIR / method).glob(f"pair_{pair_id:02d}_*.npy"))
        if not matches:
            raise FileNotFoundError(f"No method output for {method}/pair_{pair_id:02d}")
        fpath = matches[0]
    motion = np.load(fpath, allow_pickle=True)
    if motion.dtype == object:
        motion = motion.item()
        if isinstance(motion, dict):
            motion = motion.get("motion", motion)
    if motion.ndim == 3 and motion.shape[-1] == 13:
        return motion[..., :3]
    if motion.ndim == 3 and motion.shape[1] == 13:
        return np.transpose(motion[:, :3, :], (2, 0, 1))
    raise ValueError(f"Unexpected method motion shape: {motion.shape} for {fpath}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="Only render the first N pairs (0 = all)")
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--figsize", type=float, nargs=2, default=(4.0, 4.0))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cond = load_cond()
    pairs = json.load(open(EVAL_PAIRS_PATH))["pairs"]
    if args.limit > 0:
        pairs = pairs[: args.limit]

    renders_root = K_COMPARE_DIR / "renders"
    renders_root.mkdir(parents=True, exist_ok=True)

    report = {"render_times": {}, "failures": [], "clip_count": 0, "render_root": str(renders_root)}
    total_start = time.time()

    for pair in pairs:
        pid = pair["pair_id"]
        src_skel = pair["source_skel"]
        tgt_skel = pair["target_skel"]
        src_fname = pair["source_fname"]

        if src_skel not in cond:
            report["failures"].append({"pair_id": pid, "reason": f"source skel {src_skel} not in cond"})
            continue
        if tgt_skel not in cond:
            report["failures"].append({"pair_id": pid, "reason": f"target skel {tgt_skel} not in cond"})
            continue

        src_parents = [int(x) for x in cond[src_skel]["parents"]]
        tgt_parents = [int(x) for x in cond[tgt_skel]["parents"]]

        # Source
        src_out = renders_root / "source" / f"pair_{pid:02d}.mp4"
        if not (args.skip_existing and src_out.exists()):
            try:
                t0 = time.time()
                src_pos = get_source_positions(src_fname, cond)
                # Trim/pad positions to valid length: source may have padding; use full
                render_motion_to_mp4(src_pos, src_parents, src_out, fps=args.fps, figsize=tuple(args.figsize))
                report["render_times"].setdefault("source", []).append(time.time() - t0)
                report["clip_count"] += 1
                if args.verbose:
                    print(f"[source pair_{pid:02d}] done in {time.time() - t0:.1f}s -> {src_out}")
            except Exception as exc:
                report["failures"].append({
                    "pair_id": pid, "method": "source", "reason": f"{type(exc).__name__}: {exc}"})
                print(f"[FAIL source pair_{pid:02d}] {exc}", file=sys.stderr)

        # Methods
        for method in args.methods:
            out_path = renders_root / method / f"pair_{pid:02d}.mp4"
            if args.skip_existing and out_path.exists():
                continue
            try:
                t0 = time.time()
                pos = get_method_positions(method, pid, src_skel, tgt_skel)
                render_motion_to_mp4(pos, tgt_parents, out_path, fps=args.fps, figsize=tuple(args.figsize))
                report["render_times"].setdefault(method, []).append(time.time() - t0)
                report["clip_count"] += 1
                if args.verbose:
                    print(f"[{method} pair_{pid:02d}] done in {time.time() - t0:.1f}s -> {out_path}")
            except Exception as exc:
                report["failures"].append({
                    "pair_id": pid, "method": method, "reason": f"{type(exc).__name__}: {exc}"})
                print(f"[FAIL {method} pair_{pid:02d}] {exc}", file=sys.stderr)

    report["total_wall_time_sec"] = time.time() - total_start
    # Summarize per-method mean render time
    report["per_method_mean_sec"] = {
        k: float(np.mean(v)) for k, v in report["render_times"].items() if v
    }
    # Drop raw lists for compactness
    report_file = renders_root / "render_report.json"
    report_to_save = {**report, "render_times": {k: len(v) for k, v in report["render_times"].items()}}
    with open(report_file, "w") as f:
        json.dump(report_to_save, f, indent=2)
    print(json.dumps(report_to_save, indent=2))


if __name__ == "__main__":
    main()
