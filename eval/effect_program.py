"""Effect Program Extractor — analytic, morphology-invariant motion descriptor.

Extracts ψ(x) ∈ R^{T×62} from a motion clip x on any skeleton:
  τ ∈ R^8:  body transport trace (centroid velocity, rotation rates, curvature, height)
  μ ∈ R^24: support occupancy (8 depth bins × 3 lateral classes)
  η ∈ R^18: graph-spectral deformation (6 eigenvalue bands × 3 axes)
  ρ ∈ R^12: phase propagation (2 temporal bands × 2 axes × 3 features)

All components are hand-designed and morphology-invariant by construction.

Usage:
    conda run -n anytop python -m eval.effect_program --verify
"""
import numpy as np
from scipy.signal import hilbert
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh


TARGET_FRAMES = 64


def build_skeleton_graph_laplacian(parents, bone_lengths):
    """Build normalized graph Laplacian of the skeleton tree.

    Returns eigenvalues and eigenvectors sorted by eigenvalue.
    """
    J = len(parents)
    adj = np.zeros((J, J))
    for j in range(J):
        p = parents[j]
        if 0 <= p < J and p != j:
            w = 1.0 / (bone_lengths[j] + 1e-6)
            adj[j, p] = w
            adj[p, j] = w
    L = laplacian(adj, normed=True)
    eigvals, eigvecs = eigh(L)
    return eigvals, eigvecs


def canonical_body_frame(positions, subtree_masses, smoothing=0.9):
    """Compute canonical body frame per frame.

    positions: [T, J, 3]
    subtree_masses: [J] — proportional to subtree total bone length
    Returns: centroids [T, 3], rotation matrices [T, 3, 3] (columns = forward, lateral, up)
    """
    T, J, _ = positions.shape
    w = subtree_masses / (subtree_masses.sum() + 1e-8)

    centroids = np.einsum('j,tjd->td', w, positions)  # [T, 3]

    # Forward axis: smoothed horizontal centroid velocity
    vel = np.zeros_like(centroids)
    vel[1:] = centroids[1:] - centroids[:-1]
    vel[0] = vel[1]
    vel_horiz = vel.copy()
    vel_horiz[:, 1] = 0  # zero out vertical

    forward = np.zeros((T, 3))
    forward[0] = vel_horiz[0]
    for t in range(1, T):
        forward[t] = smoothing * forward[t-1] + (1 - smoothing) * vel_horiz[t]

    norms = np.linalg.norm(forward, axis=1, keepdims=True)
    # Fallback for near-zero velocity: use rest-pose principal axis
    static = norms.squeeze() < 1e-6
    if static.all():
        rest_centered = positions[0] - centroids[0]
        _, _, Vt = np.linalg.svd(rest_centered, full_matrices=False)
        fallback = Vt[0]
        fallback[1] = 0
        fallback = fallback / (np.linalg.norm(fallback) + 1e-8)
        forward[:] = fallback
    else:
        forward[~static] = forward[~static] / norms[~static]
        if static.any():
            last_good = forward[~static][0]
            for t in range(T):
                if static[t]:
                    forward[t] = last_good
                else:
                    last_good = forward[t]

    up = np.array([0.0, 1.0, 0.0])
    lateral = np.cross(up, forward)
    lat_norms = np.linalg.norm(lateral, axis=1, keepdims=True)
    lateral = lateral / (lat_norms + 1e-8)

    R = np.stack([forward, lateral, np.tile(up, (T, 1))], axis=-1)  # [T, 3, 3]

    return centroids, R


def compute_subtree_masses(parents, offsets):
    """Compute subtree bone-length mass for each joint."""
    J = len(parents)
    bone_lengths = np.linalg.norm(offsets, axis=1)
    masses = bone_lengths.copy()
    # Bottom-up accumulation
    for j in reversed(range(J)):
        p = parents[j]
        if 0 <= p < J and p != j:
            masses[p] += masses[j]
    return masses


def extract_transport(positions, centroids, R, body_scale):
    """τ ∈ R^{T×8}: body transport trace, normalized by body scale."""
    T = positions.shape[0]
    tau = np.zeros((T, 8))
    s = max(body_scale, 1e-6)

    vel_world = np.zeros((T, 3))
    vel_world[1:] = centroids[1:] - centroids[:-1]
    vel_world[0] = vel_world[1]

    for t in range(T):
        v_body = R[t].T @ vel_world[t]
        tau[t, 0] = v_body[0] / s  # v_forward (scale-normalized)
        tau[t, 1] = v_body[1] / s  # v_lateral
        tau[t, 2] = v_body[2] / s  # v_up

    for t in range(1, T):
        dR = R[t].T @ R[t-1]
        tau[t, 3] = np.arctan2(dR[1, 0], dR[0, 0])   # yaw rate
        tau[t, 4] = np.arcsin(np.clip(-dR[2, 0], -1, 1))  # pitch rate
        tau[t, 5] = np.arctan2(dR[2, 1], dR[2, 2])   # roll rate
    tau[0, 3:6] = tau[1, 3:6]

    # Bounded turn rate (avoids curvature singularity at low speed)
    speed = np.sqrt(tau[:, 0]**2 + tau[:, 1]**2)
    speed_floor = 0.01  # minimum speed before curvature is damped
    heading = np.arctan2(vel_world[:, 2], vel_world[:, 0])
    dheading = np.zeros(T)
    dheading[1:] = np.diff(heading)
    dheading = np.unwrap(dheading)
    tau[:, 6] = dheading / np.maximum(speed, speed_floor)

    # Height normalized by body scale
    tau[:, 7] = centroids[:, 1] / s

    return tau


def extract_support_occupancy(positions, centroids, parents, body_scale,
                               n_depth_bins=8, n_lat_classes=3,
                               alpha_per_scale=10.0, beta_per_scale=5.0):
    """μ ∈ R^{T×24}: support occupancy measure, scale-normalized."""
    T, J, _ = positions.shape
    s = max(body_scale, 1e-6)
    alpha = alpha_per_scale / s
    beta = beta_per_scale / s

    geo_depth = np.zeros(J)
    for j in range(J):
        depth = 0
        current = j
        while parents[current] != current and parents[current] >= 0:
            depth += 1
            current = parents[current]
        geo_depth[j] = depth
    max_depth = geo_depth.max() + 1e-8
    geo_depth_norm = geo_depth / max_depth

    rest = positions[0]
    rest_centered = rest - rest.mean(axis=0)
    _, _, Vt = np.linalg.svd(rest_centered, full_matrices=False)
    lateral_axis = Vt[1] if np.abs(Vt[1, 2]) > np.abs(Vt[1, 0]) else Vt[2]
    lat_proj = rest_centered @ lateral_axis
    lat_sign = np.sign(lat_proj)
    lat_class = (lat_sign + 1).astype(int)

    mu = np.zeros((T, n_depth_bins * n_lat_classes))

    for t in range(T):
        heights = positions[t, :, 1]
        vel_rel = np.zeros((J, 3))
        if t > 0:
            vel_rel = positions[t] - positions[t-1] - (centroids[t] - centroids[t-1])
        vel_mag = np.linalg.norm(vel_rel, axis=1)

        q = 1.0 / (1.0 + np.exp(alpha * heights + beta * vel_mag))

        depth_bin = np.clip((geo_depth_norm * n_depth_bins).astype(int), 0, n_depth_bins - 1)
        for j in range(J):
            idx = depth_bin[j] * n_lat_classes + lat_class[j]
            mu[t, idx] += q[j]

        total = mu[t].sum()
        if total > 1e-8:
            mu[t] /= total

    return mu


def extract_spectral_deformation(positions, centroids, R, eigvals, eigvecs, rest_canonical,
                                  subtree_masses, n_bands=6):
    """η ∈ R^{T×18}: graph-spectral deformation amplitudes."""
    T, J, _ = positions.shape

    # Remove rigid transport
    Y = np.zeros((T, J, 3))
    for t in range(T):
        Y[t] = (R[t].T @ (positions[t] - centroids[t]).T).T - rest_canonical

    # Weight by subtree mass
    w = subtree_masses / (subtree_masses.sum() + 1e-8)

    # Graph Fourier transform per axis
    n_modes = min(J, eigvecs.shape[1])

    # Define eigenvalue bands (normalized by max eigenvalue)
    max_eval = eigvals[-1] + 1e-8
    band_edges = np.linspace(0, 1, n_bands + 1)

    eta = np.zeros((T, n_bands * 3))

    for t in range(T):
        for ax in range(3):
            coeffs = np.zeros(n_modes)
            for k in range(n_modes):
                coeffs[k] = np.sum(w * eigvecs[:J, k] * Y[t, :, ax])

            for b in range(n_bands):
                lo = band_edges[b] * max_eval
                hi = band_edges[b + 1] * max_eval
                mask = (eigvals[:n_modes] >= lo) & (eigvals[:n_modes] < hi)
                if mask.any():
                    eta[t, b * 3 + ax] = np.sum(coeffs[mask] ** 2)

    return eta


def extract_phase_propagation(positions, centroids, R, parents, geo_depth_norm,
                               n_depth_bins=4, n_temp_bands=2):
    """ρ ∈ R^{T×12}: phase propagation features."""
    T, J, _ = positions.shape

    # Body-frame displacement
    Y = np.zeros((T, J, 3))
    rest_canonical = (R[0].T @ (positions[0] - centroids[0]).T).T
    for t in range(T):
        Y[t] = (R[t].T @ (positions[t] - centroids[t]).T).T - rest_canonical

    # Depth-binned signals
    depth_bin = np.clip((geo_depth_norm * n_depth_bins).astype(int), 0, n_depth_bins - 1)

    binned_lat = np.zeros((T, n_depth_bins))
    binned_vert = np.zeros((T, n_depth_bins))
    counts = np.zeros(n_depth_bins)

    for j in range(J):
        b = depth_bin[j]
        binned_lat[:, b] += Y[:, j, 2]  # lateral (z in body frame)
        binned_vert[:, b] += Y[:, j, 1]  # vertical (y)
        counts[b] += 1

    for b in range(n_depth_bins):
        if counts[b] > 0:
            binned_lat[:, b] /= counts[b]
            binned_vert[:, b] /= counts[b]

    rho = np.zeros((T, n_temp_bands * 2 * 3))

    depth_centers = (np.arange(n_depth_bins) + 0.5) / n_depth_bins

    for ax_idx, signal in enumerate([binned_lat, binned_vert]):
        for band_idx in range(n_temp_bands):
            # Temporal band filtering (simple: split spectrum in half)
            for b in range(n_depth_bins):
                s = signal[:, b]
                if np.std(s) < 1e-8:
                    continue

                fft = np.fft.rfft(s)
                freqs = np.fft.rfftfreq(T)
                n_freqs = len(freqs)
                mid = n_freqs // 2

                if band_idx == 0:
                    fft_band = fft.copy()
                    fft_band[mid:] = 0
                else:
                    fft_band = fft.copy()
                    fft_band[:mid] = 0

                s_band = np.fft.irfft(fft_band, n=T)

                # Analytic signal for phase
                analytic = hilbert(s_band)
                phase = np.angle(analytic)
                amplitude = np.abs(analytic)

                # Phase-vs-depth regression (slope = propagation speed)
                # We collect phase at a reference frame (mid-clip)
                ref_frame = T // 2
                phases_at_ref = []
                for bb in range(n_depth_bins):
                    s2 = signal[:, bb]
                    if np.std(s2) < 1e-8:
                        phases_at_ref.append(0)
                        continue
                    fft2 = np.fft.rfft(s2)
                    if band_idx == 0:
                        fft2[mid:] = 0
                    else:
                        fft2[:mid] = 0
                    s2_band = np.fft.irfft(fft2, n=T)
                    a2 = hilbert(s2_band)
                    phases_at_ref.append(np.angle(a2[ref_frame]))

                phases_arr = np.array(phases_at_ref)
                # Unwrap phases for regression
                phases_arr = np.unwrap(phases_arr)

                if np.std(depth_centers) > 1e-8 and np.std(phases_arr) > 1e-8:
                    slope = np.polyfit(depth_centers, phases_arr, 1)[0]
                    corr = np.corrcoef(depth_centers, phases_arr)[0, 1]
                    power = np.mean(amplitude ** 2)
                else:
                    slope = 0
                    corr = 0
                    power = 0

                # Dominant frequency
                dom_freq = freqs[np.argmax(np.abs(fft_band[1:])) + 1] if len(fft_band) > 1 else 0

                base_idx = (ax_idx * n_temp_bands + band_idx) * 3
                rho[T // 2, base_idx] = slope
                rho[T // 2, base_idx + 1] = abs(corr)
                rho[T // 2, base_idx + 2] = power

    # Broadcast scalar phase features across time (they're clip-level)
    for i in range(rho.shape[1]):
        rho[:, i] = rho[T // 2, i]

    return rho


def extract_effect_program(positions, parents, offsets):
    """Extract full effect program ψ ∈ R^{T×62} from global joint positions.

    positions: [T, J, 3] — global joint positions (denormalized)
    parents: [J] — parent indices
    offsets: [J, 3] — rest-pose bone offsets

    Returns: dict with keys 'tau', 'mu', 'eta', 'rho', 'psi' (concatenated)
    """
    T_orig, J, _ = positions.shape

    # Resample to TARGET_FRAMES
    if T_orig != TARGET_FRAMES:
        indices = np.linspace(0, T_orig - 1, TARGET_FRAMES).astype(int)
        positions = positions[indices]
    T = TARGET_FRAMES

    bone_lengths = np.linalg.norm(offsets, axis=1)
    subtree_masses = compute_subtree_masses(parents, offsets)
    centroids, R = canonical_body_frame(positions, subtree_masses)

    rest_canonical = (R[0].T @ (positions[0] - centroids[0]).T).T

    # Geodesic depth
    geo_depth = np.zeros(J)
    for j in range(J):
        depth = 0
        current = j
        while parents[current] != current and parents[current] >= 0:
            depth += 1
            current = parents[current]
        geo_depth[j] = depth
    geo_depth_norm = geo_depth / (geo_depth.max() + 1e-8)

    eigvals, eigvecs = build_skeleton_graph_laplacian(parents, bone_lengths)

    # Body scale = mean bone length (for scale normalization)
    body_scale = float(bone_lengths[bone_lengths > 1e-8].mean()) if (bone_lengths > 1e-8).any() else 1.0

    tau = extract_transport(positions, centroids, R, body_scale)
    mu = extract_support_occupancy(positions, centroids, parents, body_scale)
    eta = extract_spectral_deformation(positions, centroids, R, eigvals, eigvecs,
                                        rest_canonical, subtree_masses)
    rho = extract_phase_propagation(positions, centroids, R, parents, geo_depth_norm)

    psi = np.concatenate([tau, mu, eta, rho], axis=1)
    assert psi.shape == (TARGET_FRAMES, 62), f"Expected ({TARGET_FRAMES}, 62), got {psi.shape}"

    return {
        'tau': tau,  # [T, 8]
        'mu': mu,    # [T, 24]
        'eta': eta,  # [T, 18]
        'rho': rho,  # [T, 12]
        'psi': psi,  # [T, 62]
    }


def verify_on_truebones():
    """Verify effect extraction on representative Truebones clips."""
    import os
    from os.path import join as pjoin
    from data_loaders.truebones.truebones_utils.get_opt import get_opt
    from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np
    import json

    opt = get_opt(0)
    cond_dict = np.load(opt.cond_file, allow_pickle=True).item()
    with open('dataset/truebones/zoo/truebones_processed/motion_labels.json') as f:
        label_map = json.load(f)

    skeletons = ['Horse', 'Jaguar', 'Alligator', 'Anaconda', 'Parrot']
    skeletons = [s for s in skeletons if s in cond_dict]

    results = {}
    for skel in skeletons:
        info = cond_dict[skel]
        n_joints = len(info['joints_names'])
        parents = info['parents'][:n_joints]
        offsets_raw = info['offsets'][:n_joints]
        mean = info['mean'][:n_joints]
        std = info['std'][:n_joints]

        motion_files = sorted(f for f in os.listdir(opt.motion_dir)
                              if f.startswith(f'{skel}_'))[:3]

        for mf in motion_files:
            raw = np.load(pjoin(opt.motion_dir, mf))
            motion_denorm = raw[:, :n_joints] * (std + 1e-6) + mean
            positions = recover_from_bvh_ric_np(motion_denorm)

            try:
                eff = extract_effect_program(positions, parents, offsets_raw)
                psi = eff['psi']
                tau_range = eff['tau'].ptp(axis=0).mean()
                mu_entropy = -np.sum(eff['mu'] * np.log(eff['mu'] + 1e-8), axis=1).mean()
                eta_energy = eff['eta'].sum(axis=1).mean()
                rho_coherence = eff['rho'][:, 1::3].mean()

                print(f"  {skel}/{mf}: psi={psi.shape} tau_range={tau_range:.4f} "
                      f"mu_entropy={mu_entropy:.3f} eta_energy={eta_energy:.4f} "
                      f"rho_coherence={rho_coherence:.4f}")
                results[f'{skel}/{mf}'] = {
                    'shape': list(psi.shape),
                    'tau_range': float(tau_range),
                    'mu_entropy': float(mu_entropy),
                    'eta_energy': float(eta_energy),
                    'rho_coherence': float(rho_coherence),
                    'has_nan': bool(np.isnan(psi).any()),
                    'has_inf': bool(np.isinf(psi).any()),
                }
            except Exception as e:
                print(f"  {skel}/{mf}: FAILED — {e}")
                import traceback
                traceback.print_exc()
                results[f'{skel}/{mf}'] = {'error': str(e)}

    print(f"\n{'='*50}")
    print(f"Verified {len(results)} clips across {len(skeletons)} skeletons")
    n_ok = sum(1 for r in results.values() if 'error' not in r and not r.get('has_nan'))
    n_fail = len(results) - n_ok
    print(f"  OK: {n_ok}, Failed/NaN: {n_fail}")

    if n_ok > 0:
        ok_results = {k: v for k, v in results.items() if 'error' not in v}
        print(f"\n  Tau range:  min={min(v['tau_range'] for v in ok_results.values()):.4f}  "
              f"max={max(v['tau_range'] for v in ok_results.values()):.4f}")
        print(f"  Mu entropy: min={min(v['mu_entropy'] for v in ok_results.values()):.3f}  "
              f"max={max(v['mu_entropy'] for v in ok_results.values()):.3f}")
        print(f"  Eta energy: min={min(v['eta_energy'] for v in ok_results.values()):.4f}  "
              f"max={max(v['eta_energy'] for v in ok_results.values()):.4f}")

    return results


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--verify', action='store_true')
    args = p.parse_args()

    if args.verify:
        verify_on_truebones()
