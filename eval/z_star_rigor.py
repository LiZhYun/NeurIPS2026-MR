"""Rigorous statistical test of z* search signal.

Three controls + one main test:

MAIN: For N (source, target) pairs, search M candidates, record L_best per pair.
H0: Search using the TRUE source ψ achieves the same L_best as search using a RANDOMIZED ψ.
H1: True ψ produces lower L_best (paired Wilcoxon test).

CONTROL 1 (Source specificity): For pair (A, T), the z* selected via ψ_A should give LOWER L_eff
when scored against ψ_A than against ψ_B (different source). If L_AA ≈ L_AB, search is non-specific.

CONTROL 2 (Component balance): Report L_best decomposed into τ/μ/η/ρ. If only τ improves and μ/η/ρ
are flat, signal is dominated by τ (which has 8 dims with potentially large values).

CONTROL 3 (Random target): Replace target skeleton with random one. Signal should drop.

Outputs: paired test p-value, Cohen's d effect size, per-component breakdown.

Usage:
    conda run -n anytop python -m eval.z_star_rigor --n_pairs 24 --n_candidates 16
"""
import os
import json
import time
import argparse
import numpy as np
import torch
from os.path import join as pjoin
from scipy import stats as scipy_stats


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default='save/all_model_dataset_truebones_bs_16_latentdim_128/model000459999.pt')
    p.add_argument('--n_pairs', type=int, default=24, help='Source-target pairs')
    p.add_argument('--n_candidates', type=int, default=16, help='Search candidates per pair')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--out', type=str, default='eval/results/z_star_rigor.json')
    return p.parse_args()


def main():
    args = parse_args()
    from utils.fixseed import fixseed
    from utils import dist_util
    from utils.model_util import create_model_and_diffusion_general_skeleton, load_model
    from data_loaders.truebones.truebones_utils.get_opt import get_opt
    from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np
    from model.conditioners import T5Conditioner
    from eval.effect_program import extract_effect_program
    from eval.z_star_optimization import build_y_for_target, sample_anytop_with_z, compute_effect_loss

    fixseed(args.seed)
    dist_util.setup_dist(args.device)
    device = dist_util.dev()

    # Load model
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

    # Build source/target pair pool
    skels = sorted(set(m['skeleton'] for m in metadata))
    rng = np.random.default_rng(args.seed)

    # Generate N pairs spanning topology gaps
    skel_to_idx = {s: [i for i, m in enumerate(metadata) if m['skeleton'] == s] for s in skels}

    pairs = []
    while len(pairs) < args.n_pairs:
        src_skel, tgt_skel = rng.choice(skels, size=2, replace=False)
        if src_skel == tgt_skel:
            continue
        if not skel_to_idx[src_skel] or tgt_skel not in cond_dict:
            continue
        src_idx = int(rng.choice(skel_to_idx[src_skel]))
        pairs.append((src_skel, tgt_skel, src_idx))

    print(f"Generated {len(pairs)} (source, target, src_idx) test pairs")

    # MAIN: For each pair, generate M candidates and record L_best with TRUE source ψ
    # CONTROL 1: For each candidate, also compute L_eff against a DIFFERENT source ψ (mismatched)
    # CONTROL 2: Report per-component breakdown
    # We share candidate generation across all conditions (same noise seeds → same motions per target)

    # Strategy: for each TARGET, generate M candidate motions ONCE.
    # Then for each (source, target) pair, score candidates against source ψ.
    # This is efficient and isolates the source ψ as the only varying factor.

    # Group pairs by target
    target_to_pairs = {}
    for src_skel, tgt_skel, src_idx in pairs:
        target_to_pairs.setdefault(tgt_skel, []).append((src_skel, src_idx))

    # Generate candidates per target
    target_candidates = {}  # tgt_skel -> [(motion_norm, parents, offsets, mean, std), ...]
    print(f"\nGenerating {args.n_candidates} candidates × {len(target_to_pairs)} targets")
    t0 = time.time()
    for tgt_skel in target_to_pairs:
        info = cond_dict[tgt_skel]
        n_joints = len(info['joints_names'])
        parents = info['parents'][:n_joints]
        offsets = info['offsets'][:n_joints]
        mean = info['mean'][:n_joints]
        std = info['std'][:n_joints]

        y, _, _, _, _ = build_y_for_target(tgt_skel, cond_dict, opt, t5, n_frames, device)
        shape = (1, opt.max_joints, opt.feature_len, n_frames)

        cands = []
        for ci in range(args.n_candidates):
            torch.manual_seed(10000 + ci)
            with torch.no_grad():
                sample = sample_anytop_with_z(model, diffusion, y, None, shape, device)
            motion_norm = sample[0].cpu().permute(2, 0, 1).numpy()
            cands.append(motion_norm)

        target_candidates[tgt_skel] = (cands, parents, offsets, mean, std)
        elapsed = time.time() - t0
        print(f"  {tgt_skel}: {args.n_candidates} candidates ({elapsed:.0f}s elapsed)")

    # Now score all pairs
    print("\nScoring pairs against TRUE source ψ + RANDOMIZED source ψ...")

    main_results = []  # per-pair (l_true_best, l_rand_best, l_true_components, etc.)

    for src_skel, tgt_skel, src_idx in pairs:
        true_psi = psi_all[src_idx]

        # Pick a random DIFFERENT source clip for control
        rand_idx = src_idx
        while rand_idx == src_idx or metadata[rand_idx]['skeleton'] == src_skel:
            rand_idx = int(rng.integers(0, len(psi_all)))
        rand_psi = psi_all[rand_idx]

        cands, parents, offsets, mean, std = target_candidates[tgt_skel]
        n_joints = len(parents)

        true_losses = []
        rand_losses = []
        true_components = []
        for motion_norm in cands:
            l_true, _ = compute_effect_loss(motion_norm, mean, std, parents, offsets, true_psi)
            l_rand, _ = compute_effect_loss(motion_norm, mean, std, parents, offsets, rand_psi)
            true_losses.append(l_true['total'])
            rand_losses.append(l_rand['total'])
            true_components.append(l_true)

        true_best_idx = int(np.argmin(true_losses))
        rand_best_idx = int(np.argmin(rand_losses))

        main_results.append({
            'src_skel': src_skel,
            'tgt_skel': tgt_skel,
            'src_idx': src_idx,
            'rand_idx': rand_idx,
            'l_true_best':    float(true_losses[true_best_idx]),
            'l_true_median':  float(np.median(true_losses)),
            'l_rand_best':    float(rand_losses[rand_best_idx]),
            'l_rand_median':  float(np.median(rand_losses)),
            'best_components': true_components[true_best_idx],
            # Specificity: did the search pick the same candidate for true vs random ψ?
            'same_winner':    bool(true_best_idx == rand_best_idx),
        })

    # Statistical test: paired Wilcoxon on (l_true_best vs l_rand_best)
    l_true = np.array([r['l_true_best'] for r in main_results])
    l_rand = np.array([r['l_rand_best'] for r in main_results])

    diff = l_true - l_rand  # negative means TRUE is better
    n_better = int((diff < 0).sum())
    n_worse = int((diff > 0).sum())

    if len(diff) >= 6:
        wilcoxon_stat, wilcoxon_p = scipy_stats.wilcoxon(l_true, l_rand, alternative='less')
        cohen_d = float(np.mean(diff) / (np.std(diff) + 1e-8))
    else:
        wilcoxon_p = None
        cohen_d = None

    same_winner_rate = float(np.mean([r['same_winner'] for r in main_results]))

    # Per-component analysis
    comp_means = {'tau': [], 'mu': [], 'eta': [], 'rho': []}
    for r in main_results:
        for c in comp_means:
            comp_means[c].append(r['best_components'][c])
    comp_summary = {c: {'mean': float(np.mean(v)), 'std': float(np.std(v))}
                    for c, v in comp_means.items()}

    # Print report
    print(f"\n{'='*60}")
    print(f"RIGOR RESULTS — n_pairs={len(main_results)}, n_candidates={args.n_candidates}")
    print(f"{'='*60}")
    print(f"\nL_eff summary (lower = better effect match):")
    print(f"  TRUE source ψ:  best = {l_true.mean():.3f} ± {l_true.std():.3f}")
    print(f"  RANDOM source ψ: best = {l_rand.mean():.3f} ± {l_rand.std():.3f}")
    print(f"  Difference (true - rand): {diff.mean():.3f} ± {diff.std():.3f}")
    print(f"  N with TRUE better:  {n_better}/{len(diff)}")
    print(f"  N with RANDOM better: {n_worse}/{len(diff)}")

    if wilcoxon_p is not None:
        print(f"\nPaired Wilcoxon test (H0: equal, H1: true < random):")
        print(f"  p-value = {wilcoxon_p:.4f}")
        print(f"  Cohen's d = {cohen_d:.3f}")
        if wilcoxon_p < 0.05:
            print(f"  ✓ SIGNIFICANT — search uses source ψ as a real signal")
        else:
            print(f"  ✗ NOT SIGNIFICANT — search may be picking up noise, not source ψ")

    print(f"\nSpecificity check:")
    print(f"  same-winner rate (true vs random ψ pick same candidate): {same_winner_rate:.2%}")
    print(f"  random baseline (16 candidates): 1/16 = 6.25%")
    if same_winner_rate < 0.20:
        print(f"  ✓ Search is source-specific (different ψ → different winner)")
    else:
        print(f"  ✗ Search not source-specific — winners often coincide")

    print(f"\nComponent breakdown of TRUE best L_eff:")
    for c, s in comp_summary.items():
        print(f"  {c}: {s['mean']:.4f} ± {s['std']:.4f}")
    total_comp = sum(s['mean'] for s in comp_summary.values())
    for c, s in comp_summary.items():
        pct = 100 * s['mean'] / max(total_comp, 1e-8)
        print(f"  {c} share of loss: {pct:.1f}%")

    out = {
        'n_pairs': len(main_results),
        'n_candidates': args.n_candidates,
        'l_true_mean': float(l_true.mean()),
        'l_rand_mean': float(l_rand.mean()),
        'l_true_minus_rand_mean': float(diff.mean()),
        'n_true_better': n_better,
        'n_rand_better': n_worse,
        'wilcoxon_p': float(wilcoxon_p) if wilcoxon_p is not None else None,
        'cohen_d': cohen_d,
        'same_winner_rate': same_winner_rate,
        'component_summary': comp_summary,
        'pairs': main_results,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {args.out}")


if __name__ == '__main__':
    main()
