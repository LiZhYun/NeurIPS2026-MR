"""Task-space quotient Q(x) extractor for Idea K (Phase 4.5, 2026-04-14).

Computes five morphology-agnostic, causal task-space variables from a motion clip.
This is the DELIBERATELY LOW-DIMENSIONAL replacement for ψ (which pilot D showed
is dominated 5x by action label and does not carry fine-grained cross-skeleton
signal). Q is designed to be low-dim, hand-designed, and CAUSAL for the motion.

Components:
  1. COM path              [T, 3]   centre-of-mass trajectory (world frame)
  2. Heading velocity      [T]      body-frame forward speed
  3. Contact schedule      [T, C]   per-frame contacts per named contact group
                                    aggregate version: C = 1 (total contact count)
                                    grouped version:   C >= 1 from contact_groups.json
  4. Cadence               scalar   dominant frequency of aggregate contact schedule
  5. Limb-usage distribution [K]    normalised kinetic energy per kinematic-chain

Reuses:
  - canonical_body_frame (eval/effect_program.py) for centroids + R
  - recover_from_bvh_ric_np (data_loaders/.../motion_process.py) for positions

Usage:
  python -m eval.quotient_extractor --demo     # run on 3 sample clips
  python -m eval.quotient_extractor --build     # build idea-stage/quotient_cache.npz
"""
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import argparse
import json
import os
from os.path import join as pjoin
from pathlib import Path

import numpy as np

ROOT = Path(str(PROJECT_ROOT))


def load_cond(cond_path=None):
    if cond_path is None:
        from data_loaders.truebones.truebones_utils.param_utils import DATASET_DIR
        cond_path = pjoin(DATASET_DIR, 'cond.npy')
    return np.load(cond_path, allow_pickle=True).item()


def load_motion_positions(fname, motion_dir=None, n_joints=None):
    """Load a motion .npy and convert to (positions [T, J, 3], contacts [T, J])."""
    from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np
    if motion_dir is None:
        from data_loaders.truebones.truebones_utils.param_utils import DATASET_DIR
        motion_dir = pjoin(DATASET_DIR, 'motions')
    m = np.load(pjoin(motion_dir, fname))  # [T, J, 13]
    if n_joints is not None and m.shape[1] > n_joints:
        m = m[:, :n_joints]
    contacts = (m[..., 12] > 0.5).astype(np.int8)  # [T, J]
    positions = recover_from_bvh_ric_np(m.astype(np.float32))  # [T, J, 3]
    return positions, contacts


def compute_subtree_masses(parents, offsets):
    J = len(parents)
    bone_lengths = np.linalg.norm(offsets, axis=1)
    masses = bone_lengths.copy()
    for j in reversed(range(J)):
        p = parents[j]
        if 0 <= p < J and p != j:
            masses[p] += masses[j]
    return masses


def body_scale(offsets):
    """Body scale = max joint-to-root distance in rest pose."""
    J = offsets.shape[0]
    # Reconstruct rest-pose joint positions
    positions = np.zeros_like(offsets)
    # Assume parent chain is well-ordered (parents[j] < j for j > 0)
    for j in range(1, J):
        positions[j] = positions[max(j-1, 0)] + offsets[j]  # crude but OK for scale estimate
    # Better: recursive resolve
    positions2 = np.zeros_like(offsets)
    # Placeholder; for K the scale just needs to be monotone in body size
    # Use the sum of all bone lengths as a rough proxy
    return float(np.linalg.norm(offsets, axis=1).sum() + 1e-6)


def compute_heading_velocity(centroids, R, fps=30):
    """Body-frame forward speed magnitude, per frame."""
    T = centroids.shape[0]
    vel = np.zeros_like(centroids)
    vel[1:] = (centroids[1:] - centroids[:-1]) * fps
    vel[0] = vel[1]
    # R columns: [forward, lateral, up]. Project onto forward axis.
    forward_speed = np.einsum('td,td->t', vel, R[:, :, 0])
    return forward_speed


def compute_contact_schedule_aggregate(contacts):
    """Aggregate contact schedule: total contact count per frame, normalised by J."""
    T, J = contacts.shape
    return contacts.sum(axis=1).astype(np.float32) / max(J, 1)


def compute_contact_schedule_grouped(contacts, groups):
    """Per-group contact schedule.

    contacts: [T, J]
    groups: dict {group_name: list[joint_index]}
    Returns: [T, C] contact-fraction per group, and group_names list
    """
    T, J = contacts.shape
    names = sorted(groups.keys())
    C = len(names)
    sched = np.zeros((T, C), dtype=np.float32)
    for i, name in enumerate(names):
        idxs = [j for j in groups[name] if 0 <= j < J]
        if idxs:
            sched[:, i] = contacts[:, idxs].mean(axis=1)
    return sched, names


def compute_cadence(contact_schedule, fps=30, min_cycle=0.25, max_cycle=4.0):
    """Dominant temporal frequency of the contact schedule.

    contact_schedule: [T] or [T, C] → we take sum across groups if C>1.
    Returns: scalar cadence in cycles/second (Hz).
    """
    if contact_schedule.ndim > 1:
        s = contact_schedule.sum(axis=1)
    else:
        s = contact_schedule
    s = s - s.mean()
    if s.std() < 1e-6:
        return 0.0
    T = len(s)
    fft = np.fft.rfft(s)
    freqs = np.fft.rfftfreq(T, d=1.0 / fps)
    # Filter to realistic cadence range
    lo = 1.0 / max_cycle  # e.g. 0.25 Hz
    hi = 1.0 / min_cycle  # e.g. 4 Hz
    mask = (freqs >= lo) & (freqs <= hi)
    if not mask.any():
        return 0.0
    power = np.abs(fft) ** 2
    power_mask = power[mask]
    freq_mask = freqs[mask]
    peak_idx = np.argmax(power_mask)
    return float(freq_mask[peak_idx])


def compute_limb_usage(positions, kinematic_chains, fps=30):
    """Normalised kinetic energy per kinematic chain.

    positions: [T, J, 3]
    kinematic_chains: list[list[int]] (from cond)
    Returns: [K] summing to 1, and chain lengths
    """
    T, J, _ = positions.shape
    vel = np.zeros_like(positions)
    vel[1:] = (positions[1:] - positions[:-1]) * fps
    vel[0] = vel[1]
    ke = 0.5 * (vel ** 2).sum(axis=-1)  # [T, J]
    ke_total_per_joint = ke.mean(axis=0)  # [J]
    K = len(kinematic_chains)
    energy = np.zeros(K)
    for k, chain in enumerate(kinematic_chains):
        idxs = [j for j in chain if 0 <= j < J]
        if idxs:
            energy[k] = ke_total_per_joint[idxs].mean()
    total = energy.sum() + 1e-12
    return energy / total


def extract_quotient(fname, cond, contact_groups=None, n_joints=None,
                     motion_dir=None, fps=30):
    """Extract the 5-component quotient Q(x) for a motion clip.

    Returns a dict:
      com_path         [T, 3]  world-frame COM (body-scale-normalised + gravity-up aligned)
      heading_vel      [T]     body-frame forward speed (Hz, body-scale-normalised)
      contact_sched    [T, C]  if contact_groups provided, else [T] aggregate
      contact_group_names  list[str] if grouped, else None
      cadence          scalar (Hz)
      limb_usage       [K]     normalised kinetic energy per kinematic chain
      body_scale       scalar  for normalisation by consumer
      n_joints         int
    """
    parents = cond['parents']
    offsets = cond['offsets']
    chains = cond['kinematic_chains']
    J = offsets.shape[0]

    positions, contacts = load_motion_positions(fname, motion_dir=motion_dir, n_joints=J)
    T = positions.shape[0]

    subtree = compute_subtree_masses(parents, offsets)
    # We import canonical_body_frame lazily
    from eval.effect_program import canonical_body_frame
    centroids, R = canonical_body_frame(positions, subtree)

    scale = body_scale(offsets)

    # Component 1: COM path — body-scale-normalised
    com_path = (centroids - centroids[0:1]) / scale  # anchor to zero at frame 0

    # Component 2: heading velocity — body-scale-normalised
    heading_vel = compute_heading_velocity(centroids, R, fps=fps) / scale

    # Component 3: contact schedule
    if contact_groups is not None and cond['object_type'] in contact_groups:
        sched, names = compute_contact_schedule_grouped(contacts, contact_groups[cond['object_type']])
    else:
        sched = compute_contact_schedule_aggregate(contacts)
        names = None

    # Component 4: cadence
    cadence = compute_cadence(sched, fps=fps)

    # Component 5: limb-usage
    limb_usage = compute_limb_usage(positions, chains, fps=fps)

    return {
        'com_path': com_path.astype(np.float32),
        'heading_vel': heading_vel.astype(np.float32),
        'contact_sched': sched.astype(np.float32),
        'contact_group_names': names,
        'cadence': float(cadence),
        'limb_usage': limb_usage.astype(np.float32),
        'body_scale': float(scale),
        'n_joints': int(J),
        'n_frames': int(T),
    }


def demo():
    """Run on 3 clips from 3 different morphology families."""
    cond = load_cond()
    demos = [
        ('Horse___LandRun_448.npy', 'Horse'),
        ('Chicken___Walk_189.npy', 'Chicken'),  # may not exist
        ('Anaconda___SlowForward_32.npy', 'Anaconda'),  # may not exist
    ]
    # Find any available clips if these don't exist
    from data_loaders.truebones.truebones_utils.param_utils import DATASET_DIR
    motion_dir = pjoin(DATASET_DIR, 'motions')
    available = set(os.listdir(motion_dir))

    # pick fallback clips if needed
    for skel in ['Horse', 'Chicken', 'Anaconda', 'Bear', 'Cat']:
        matches = [f for f in available if f.startswith(skel + '___')]
        if matches:
            demos.append((matches[0], skel))
    # de-duplicate by skel
    seen = set()
    filtered = []
    for f, s in demos:
        if s not in seen and f in available:
            filtered.append((f, s))
            seen.add(s)
        if len(filtered) >= 3:
            break

    print(f"\n{'='*60}\nTask-space quotient Q(x) demo — {len(filtered)} clips\n{'='*60}")
    for fname, skel in filtered:
        if skel not in cond:
            print(f"  skip {fname}: no cond for {skel}")
            continue
        print(f"\n--- {fname} (skel={skel}, J={cond[skel]['offsets'].shape[0]}) ---")
        q = extract_quotient(fname, cond[skel])
        print(f"  COM path:      shape={q['com_path'].shape}, range=[{q['com_path'].min():.3f}, {q['com_path'].max():.3f}]")
        print(f"  heading vel:   shape={q['heading_vel'].shape}, mean={q['heading_vel'].mean():.3f}, max={q['heading_vel'].max():.3f}")
        print(f"  contact sched: shape={q['contact_sched'].shape}, mean={q['contact_sched'].mean():.3f}, groups={q['contact_group_names']}")
        print(f"  cadence:       {q['cadence']:.3f} Hz")
        print(f"  limb_usage:    shape={q['limb_usage'].shape}, top3 chains: {np.argsort(-q['limb_usage'])[:3].tolist()}")
        print(f"  body_scale:    {q['body_scale']:.3f}")
        print(f"  n_joints: {q['n_joints']}, n_frames: {q['n_frames']}")


def build_cache(out_path=None, max_clips=None):
    """Compute Q for every clip in the dataset and save a cache."""
    cond = load_cond()
    from data_loaders.truebones.truebones_utils.param_utils import DATASET_DIR
    motion_dir = pjoin(DATASET_DIR, 'motions')
    meta_path = pjoin(ROOT, 'eval/results/effect_cache/clip_metadata.json')
    with open(meta_path) as f:
        meta = json.load(f)

    if max_clips:
        meta = meta[:max_clips]

    if out_path is None:
        out_path = ROOT / 'idea-stage/quotient_cache.npz'

    all_com_path = []       # ragged — save as list
    all_heading_vel = []
    all_contact_sched = []
    all_cadence = []
    all_limb_usage = []
    kept_meta = []
    for i, m in enumerate(meta):
        if i % 100 == 0:
            print(f"  {i}/{len(meta)}")
        if m['skeleton'] not in cond:
            continue
        try:
            q = extract_quotient(m['fname'], cond[m['skeleton']], motion_dir=motion_dir)
        except Exception as e:
            print(f"  skip {m['fname']}: {e}")
            continue
        all_com_path.append(q['com_path'])
        all_heading_vel.append(q['heading_vel'])
        all_contact_sched.append(q['contact_sched'])
        all_cadence.append(q['cadence'])
        all_limb_usage.append(q['limb_usage'])
        kept_meta.append(m)

    np.savez(
        out_path,
        com_path=np.array(all_com_path, dtype=object),
        heading_vel=np.array(all_heading_vel, dtype=object),
        contact_sched=np.array(all_contact_sched, dtype=object),
        cadence=np.array(all_cadence, dtype=np.float32),
        limb_usage=np.array(all_limb_usage, dtype=object),
        meta=np.array(kept_meta, dtype=object),
    )
    print(f"Saved: {out_path}  ({len(kept_meta)} clips)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--demo', action='store_true')
    p.add_argument('--build', action='store_true')
    p.add_argument('--max_clips', type=int, default=None)
    args = p.parse_args()
    if args.demo:
        demo()
    elif args.build:
        build_cache(max_clips=args.max_clips)
    else:
        p.print_help()


if __name__ == '__main__':
    main()
