"""Render MP4 demos of the synthetic 2x2 toy world.

Produces four videos under save/synthetic_2x2/demos/:
  1. actions_overview.mp4   - all 6 actions on a J=14 chain (2x3 grid)
  2. skeleton_scale.mp4     - locomotion on J in {8, 14, 22} (1x3 grid)
  3. oracle_transport.mp4   - source on J=8 vs oracle T*(x_a) on J=22, fly action (1x2)
  4. instance_noise.mp4     - 5 instances of (J=14, fly) with different phase+amp (1x5)

Usage:
  python -m eval.synthetic_2x2.render_demos
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.synthetic_2x2.build_synth_dataset import (
    ACTIONS, JOINTS_PER_SKEL, T_FRAMES,
    action_trajectory, true_transport,
)

OUT_DIR = PROJECT_ROOT / 'save/synthetic_2x2/demos'
OUT_DIR.mkdir(parents=True, exist_ok=True)

FPS = 24
DPI = 110


# ---------- Plot helpers ----------

def axes_limits(clips):
    """Common axis limits across a list of [T, J, 3] clips."""
    flat = np.concatenate([c.reshape(-1, 3) for c in clips], axis=0)
    pad = 0.5
    return (flat[:, 0].min() - pad, flat[:, 0].max() + pad,
            flat[:, 1].min() - pad, flat[:, 1].max() + pad)


def setup_panel(ax, xlim, ylim, title):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)


def draw_skeleton(ax, frame, color, lw=2.0, ms=5.0, alpha=1.0):
    """Return the line and scatter artists for one skeleton frame ([J, 3])."""
    line, = ax.plot(frame[:, 0], frame[:, 1], '-', color=color, lw=lw, alpha=alpha)
    scat = ax.scatter(frame[:, 0], frame[:, 1], s=ms ** 2, c=color, alpha=alpha,
                      zorder=3, edgecolors='white', linewidths=0.5)
    return line, scat


def update_skeleton(line, scat, frame):
    line.set_data(frame[:, 0], frame[:, 1])
    scat.set_offsets(frame[:, :2])


# ---------- Video 1: all 6 actions on J=14 ----------

def render_actions_overview():
    J = 14
    clips = [action_trajectory(a, J, T_FRAMES, instance_seed=42, add_instance_noise=False)
             for a in ACTIONS]
    xlim_y = axes_limits(clips)
    xlim, ylim = xlim_y[0:2], xlim_y[2:4]

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.5), dpi=DPI)
    axes = axes.flatten()
    artists = []
    for ax, clip, name in zip(axes, clips, ACTIONS):
        setup_panel(ax, xlim, ylim, f'{name}  (J={J})')
        ln, sc = draw_skeleton(ax, clip[0], color='#1f6feb', lw=2.0, ms=5.5)
        artists.append((ln, sc, clip))

    fig.suptitle('Synthetic toy world: 6 actions on a 14-joint chain', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    def update(i):
        out = []
        for ln, sc, clip in artists:
            update_skeleton(ln, sc, clip[i])
            out.extend([ln, sc])
        return out

    a = anim.FuncAnimation(fig, update, frames=T_FRAMES, interval=1000 / FPS, blit=False)
    out = OUT_DIR / 'actions_overview.mp4'
    a.save(out, writer=anim.FFMpegWriter(fps=FPS, bitrate=2400))
    plt.close(fig)
    return out


# ---------- Video 2: locomotion on J=8, 14, 22 ----------

def render_skeleton_scale():
    Js = [8, 14, 22]
    clips = [action_trajectory('locomotion', J, T_FRAMES, instance_seed=42, add_instance_noise=False)
             for J in Js]
    # Per-skeleton axis limits since chain length differs
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.8), dpi=DPI)
    artists = []
    for ax, clip, J in zip(axes, clips, Js):
        # Use a fat margin so all three appear at similar visual scale
        xlim_y = axes_limits([clip])
        setup_panel(ax, xlim_y[0:2], xlim_y[2:4], f'locomotion, J={J}')
        ln, sc = draw_skeleton(ax, clip[0], color='#1f6feb', lw=2.0, ms=5.5)
        artists.append((ln, sc, clip))

    fig.suptitle('Same action across three skeleton sizes', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    def update(i):
        out = []
        for ln, sc, clip in artists:
            update_skeleton(ln, sc, clip[i])
            out.extend([ln, sc])
        return out

    a = anim.FuncAnimation(fig, update, frames=T_FRAMES, interval=1000 / FPS, blit=False)
    out = OUT_DIR / 'skeleton_scale.mp4'
    a.save(out, writer=anim.FFMpegWriter(fps=FPS, bitrate=2400))
    plt.close(fig)
    return out


# ---------- Video 3: oracle transport, fly action, J=8 -> J=22 ----------

def render_oracle_transport():
    J_a, J_b = 8, 22
    action = 'fly'
    x_a = action_trajectory(action, J_a, T_FRAMES, instance_seed=7, add_instance_noise=True)
    x_b_oracle = true_transport(x_a, J_b, action)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 5.0), dpi=DPI)

    xlim_a = axes_limits([x_a])
    xlim_b = axes_limits([x_b_oracle])
    # Use a shared scale so the chain-length difference is honest
    xmin = min(xlim_a[0], xlim_b[0]); xmax = max(xlim_a[1], xlim_b[1])
    ymin = min(xlim_a[2], xlim_b[2]); ymax = max(xlim_a[3], xlim_b[3])
    setup_panel(axes[0], (xmin, xmax), (ymin, ymax), f'source on skel-A (J={J_a})')
    setup_panel(axes[1], (xmin, xmax), (ymin, ymax), f'oracle T*(x_a) on skel-B (J={J_b})')

    ln_a, sc_a = draw_skeleton(axes[0], x_a[0], color='#0a7f3f', lw=2.2, ms=5.5)
    ln_b, sc_b = draw_skeleton(axes[1], x_b_oracle[0], color='#b5651d', lw=2.2, ms=5.5)

    fig.suptitle(f'Oracle transport: {action}, J={J_a} -> J={J_b}', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    def update(i):
        update_skeleton(ln_a, sc_a, x_a[i])
        update_skeleton(ln_b, sc_b, x_b_oracle[i])
        return [ln_a, sc_a, ln_b, sc_b]

    a = anim.FuncAnimation(fig, update, frames=T_FRAMES, interval=1000 / FPS, blit=False)
    out = OUT_DIR / 'oracle_transport.mp4'
    a.save(out, writer=anim.FFMpegWriter(fps=FPS, bitrate=2400))
    plt.close(fig)
    return out


# ---------- Video 4: 5 instances of (J=14, fly) with different phase+amp ----------

def render_instance_noise():
    J = 14
    action = 'fly'
    seeds = [11, 23, 37, 51, 67]
    clips = [action_trajectory(action, J, T_FRAMES, instance_seed=s, add_instance_noise=True)
             for s in seeds]
    xlim_y = axes_limits(clips)
    xlim, ylim = xlim_y[0:2], xlim_y[2:4]

    fig, axes = plt.subplots(1, 5, figsize=(13.5, 4.0), dpi=DPI)
    artists = []
    for k, (ax, clip) in enumerate(zip(axes, clips)):
        setup_panel(ax, xlim, ylim, f'instance {k + 1}')
        ln, sc = draw_skeleton(ax, clip[0], color='#7e3ff2', lw=2.0, ms=5.0)
        artists.append((ln, sc, clip))

    fig.suptitle(f'Five dense-cell instances of ({action}, J={J}): same intent, random (phase, amp)',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    def update(i):
        out = []
        for ln, sc, clip in artists:
            update_skeleton(ln, sc, clip[i])
            out.extend([ln, sc])
        return out

    a = anim.FuncAnimation(fig, update, frames=T_FRAMES, interval=1000 / FPS, blit=False)
    out = OUT_DIR / 'instance_noise.mp4'
    a.save(out, writer=anim.FFMpegWriter(fps=FPS, bitrate=2400))
    plt.close(fig)
    return out


# ---------- Main ----------

def main():
    print(f'Output dir: {OUT_DIR}')
    paths = []
    for fn in [render_actions_overview, render_skeleton_scale,
               render_oracle_transport, render_instance_noise]:
        print(f'  rendering {fn.__name__}...')
        p = fn()
        size_kb = p.stat().st_size / 1024
        print(f'    -> {p}   ({size_kb:.1f} KB)')
        paths.append(p)

    print('\nAll demos written:')
    for p in paths:
        print(f'  {p}')


if __name__ == '__main__':
    main()
