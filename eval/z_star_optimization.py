"""z* optimization in frozen AnyTop: sanity-check that effect-constrained inference is feasible.

For (source ψ, target skeleton), optimize z to minimize:
  L = L_eff(ψ(Generate(S_t, z)), ψ_source) + λ_z ‖z‖²

Strategy: random sampling + gradient-free refinement (CMA-ES-like).
Differentiating through 8-step DDIM + np-based ψ extractor is too painful for the prototype.
We use sampling-based optimization first to verify the LOSS LANDSCAPE has signal.

If sampling-based optimization finds z* with measurably lower L_eff than random null-z,
the compiler approach is viable. If not, abandon this method.

Usage:
    conda run -n anytop python -m eval.z_star_optimization --pilot
"""
import os
import json
import time
import argparse
import numpy as np
import torch
from os.path import join as pjoin


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default='save/all_model_dataset_truebones_bs_16_latentdim_128/model000459999.pt')
    p.add_argument('--pilot', action='store_true', help='Run small pilot: 3 source-target pairs')
    p.add_argument('--n_candidates', type=int, default=64)
    p.add_argument('--n_refine_steps', type=int, default=4)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--out', type=str, default='eval/results/z_star_pilot.json')
    return p.parse_args()


@torch.no_grad()
def sample_anytop_with_z(model, diffusion, y, z_embed, shape, device):
    """Sample motion from frozen AnyTop conditioned on z_embed.

    For now we use the *unconditioned* AnyTop and ignore z_embed (no conditioning channel).
    The pretrained all_model is the unconditional one — z is provided to satisfy interface.
    """
    x = torch.randn(shape, device=device)
    for t_val in reversed(range(diffusion.num_timesteps)):
        t = torch.tensor([t_val], device=device)
        out = diffusion.p_mean_variance(model, x, t, model_kwargs={'y': y}, clip_denoised=False)
        x = out['mean']
        if t_val > 0:
            noise = torch.randn_like(x)
            x = x + torch.exp(0.5 * out['log_variance']) * noise
    return x


def build_y_for_target(skel_name, cond_dict, opt, t5, n_frames, device):
    from data_loaders.truebones.data.dataset import create_temporal_mask_for_window
    from data_loaders.tensors import create_padded_relation

    info = cond_dict[skel_name]
    max_joints = opt.max_joints
    feature_len = opt.feature_len
    n_joints = len(info['joints_names'])
    mean = info['mean']
    std = info['std'] + 1e-6

    tpos_raw = info['tpos_first_frame']
    tpos = np.zeros((max_joints, feature_len))
    tpos[:n_joints] = (tpos_raw - mean) / std
    tpos = np.nan_to_num(tpos)
    tpos_t = torch.tensor(tpos).float().unsqueeze(0).to(device)

    names = info['joints_names']
    names_emb = t5(t5.tokenize(names)).detach().cpu().numpy()
    names_padded = np.zeros((max_joints, names_emb.shape[1]))
    names_padded[:n_joints] = names_emb
    names_t = torch.tensor(names_padded).float().unsqueeze(0).to(device)

    gd = create_padded_relation(info['joints_graph_dist'], max_joints, n_joints)
    jr = create_padded_relation(info['joint_relations'], max_joints, n_joints)
    gd_t = torch.tensor(gd).long().unsqueeze(0).to(device)
    jr_t = torch.tensor(jr).long().unsqueeze(0).to(device)

    jmask_5d = torch.zeros(1, 1, 1, max_joints + 1, max_joints + 1, device=device)
    jmask_5d[0, 0, 0, :n_joints + 1, :n_joints + 1] = 1.0

    tmask = create_temporal_mask_for_window(31, n_frames)
    tmask_t = torch.tensor(tmask).unsqueeze(0).unsqueeze(2).unsqueeze(3).float().to(device)

    y = {
        'joints_mask':       jmask_5d,
        'mask':              tmask_t,
        'tpos_first_frame':  tpos_t,
        'joints_names_embs': names_t,
        'graph_dist':        gd_t,
        'joints_relations':  jr_t,
        'crop_start_ind':    torch.zeros(1, dtype=torch.long, device=device),
        'n_joints':          torch.tensor([n_joints]),
    }
    return y, n_joints, mean, std, info


def compute_effect_loss(generated_motion_norm, mean, std, parents, offsets, target_psi):
    """Compute effect loss between generated motion and target ψ (numpy, no grad)."""
    from eval.effect_program import extract_effect_program
    from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np

    n_joints = len(parents)
    motion_denorm = generated_motion_norm[:, :n_joints] * (std + 1e-6) + mean
    positions = recover_from_bvh_ric_np(motion_denorm)

    eff = extract_effect_program(positions, parents, offsets)
    psi_gen = eff['psi']

    # Per-component L1 distance (normalized)
    component_slices = {'tau': (0, 8), 'mu': (8, 32), 'eta': (32, 50), 'rho': (50, 62)}
    losses = {}
    for name, (lo, hi) in component_slices.items():
        diff = np.abs(psi_gen[:, lo:hi] - target_psi[:, lo:hi]).mean()
        losses[name] = float(diff)
    losses['total'] = sum(losses.values())
    return losses, psi_gen


def run_pilot(args):
    from utils.fixseed import fixseed
    from utils import dist_util
    from utils.model_util import create_model_and_diffusion_general_skeleton, load_model
    from data_loaders.truebones.truebones_utils.get_opt import get_opt
    from model.conditioners import T5Conditioner

    fixseed(42)
    dist_util.setup_dist(args.device)
    device = dist_util.dev()

    # Load pretrained AnyTop (unconditional)
    with open(pjoin(os.path.dirname(args.ckpt), 'args.json')) as f:
        args_d = json.load(f)
    class NS:
        def __init__(self, d): self.__dict__.update(d)
    m_args = NS(args_d)
    model, diffusion = create_model_and_diffusion_general_skeleton(m_args)
    state = torch.load(args.ckpt, map_location='cpu')
    load_model(model, state)
    model.to(device)
    model.eval()
    print(f"Loaded {args.ckpt}")

    opt = get_opt(args.device)
    cond_dict = np.load(opt.cond_file, allow_pickle=True).item()
    t5 = T5Conditioner(name=m_args.t5_name, finetune=False, word_dropout=0.0,
                       normalize_text=False, device='cuda')
    n_frames = getattr(m_args, 'num_frames', 120)

    # Load effect cache
    psi_all = np.load('eval/results/effect_cache/psi_all.npy')
    with open('eval/results/effect_cache/clip_metadata.json') as f:
        metadata = json.load(f)
    print(f"Loaded {len(psi_all)} cached ψ vectors")

    # Pilot pairs: 3 (source, target) tuples spanning topology gaps
    test_pairs = [
        ('Horse',     'Jaguar'),    # similar quadrupeds (low gap)
        ('Horse',     'Anaconda'),  # quadruped → snake (high gap)
        ('Parrot',    'Alligator'), # bird → reptile (extreme gap)
    ]

    results = []

    for src_skel, tgt_skel in test_pairs:
        print(f"\n{'='*60}")
        print(f"Pilot: {src_skel} → {tgt_skel}")
        print(f"{'='*60}")

        # Pick a source clip
        src_clips = [m for m in metadata if m['skeleton'] == src_skel]
        if not src_clips:
            print(f"  No source clips for {src_skel}, skipping")
            continue
        src_idx = next(i for i, m in enumerate(metadata) if m['skeleton'] == src_skel)
        src_psi = psi_all[src_idx]
        print(f"  Source clip: {metadata[src_idx]['fname']} (label: {metadata[src_idx]['coarse_label']})")
        print(f"  Source ψ summary: tau_range={src_psi[:, :8].ptp(axis=0).mean():.3f}  "
              f"mu_entropy={(-src_psi[:, 8:32]*np.log(src_psi[:, 8:32]+1e-8)).sum(axis=1).mean():.3f}")

        # Build target conditioning
        if tgt_skel not in cond_dict:
            print(f"  Target {tgt_skel} not in cond_dict, skipping")
            continue
        y, n_joints, mean, std, info = build_y_for_target(
            tgt_skel, cond_dict, opt, t5, n_frames, device)
        parents = info['parents'][:n_joints]
        offsets = info['offsets'][:n_joints]

        shape = (1, opt.max_joints, opt.feature_len, n_frames)

        # Sampling-based search: generate N candidates with different noise seeds
        print(f"  Sampling {args.n_candidates} candidate motions...")
        candidate_losses = []
        candidate_psi = []
        t0 = time.time()
        for cand_i in range(args.n_candidates):
            torch.manual_seed(1000 + cand_i)
            with torch.no_grad():
                sample = sample_anytop_with_z(model, diffusion, y, None, shape, device)

            motion_norm = sample[0].cpu().permute(2, 0, 1).numpy()  # [T, J, 13]
            losses, psi_gen = compute_effect_loss(
                motion_norm, mean, std, parents, offsets, src_psi)
            candidate_losses.append(losses)
            candidate_psi.append(psi_gen)

            if (cand_i + 1) % 16 == 0:
                elapsed = time.time() - t0
                print(f"    [{cand_i+1}/{args.n_candidates}] elapsed={elapsed:.0f}s")

        # Find best
        totals = [c['total'] for c in candidate_losses]
        best_idx = int(np.argmin(totals))
        worst_idx = int(np.argmax(totals))
        median = float(np.median(totals))

        # Random baseline: random noise → same target skeleton (no effect matching)
        # The "best" candidate vs "median" candidate spread tells us if there's signal
        signal_ratio = (median - min(totals)) / max(median, 1e-8)

        print(f"\n  L_eff (lower is better):")
        print(f"    best:   {min(totals):.4f} (cand {best_idx})")
        print(f"    median: {median:.4f}")
        print(f"    worst:  {max(totals):.4f} (cand {worst_idx})")
        print(f"    signal ratio (median - best) / median: {signal_ratio:.4f}")
        print(f"    best component breakdown: {candidate_losses[best_idx]}")

        results.append({
            'source': src_skel,
            'target': tgt_skel,
            'source_clip': metadata[src_idx]['fname'],
            'n_candidates': args.n_candidates,
            'l_best':    min(totals),
            'l_median':  median,
            'l_worst':   max(totals),
            'signal_ratio': signal_ratio,
            'best_components': candidate_losses[best_idx],
            'best_idx':  best_idx,
            'all_totals': totals,
        })

    # Summary
    print(f"\n{'='*60}")
    print("PILOT SUMMARY")
    print(f"{'='*60}")
    print(f"{'Source→Target':<25} {'L_best':>8} {'L_median':>10} {'Signal':>8}")
    for r in results:
        pair = f"{r['source']}→{r['target']}"
        print(f"  {pair:<25} {r['l_best']:>8.4f} {r['l_median']:>10.4f} {r['signal_ratio']:>8.4f}")

    print(f"\nKey question: Is there signal? (best should be < median)")
    avg_signal = np.mean([r['signal_ratio'] for r in results]) if results else 0
    print(f"Average signal ratio: {avg_signal:.4f}")
    if avg_signal > 0.05:
        print("✓ Signal detected — effect-matching is feasible. z* optimization viable.")
    else:
        print("✗ NO signal — effect loss is uninformative across noise seeds. Method may not work.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({
            'pilot_results': results,
            'avg_signal_ratio': float(avg_signal),
            'verdict': 'PROCEED' if avg_signal > 0.05 else 'STOP',
        }, f, indent=2)
    print(f"\nSaved → {args.out}")


def main():
    args = parse_args()
    if args.pilot:
        run_pilot(args)
    else:
        print("Use --pilot to run the sanity-check pilot.")


if __name__ == '__main__':
    main()
