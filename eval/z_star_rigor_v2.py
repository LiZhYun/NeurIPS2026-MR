"""Rigorous z* test v2 — applies all three fixes from V5 negative finding:

  1. Per-component z-score normalization of L_eff (so μ/η/ρ contribute meaningfully)
  2. Switch to CONDITIONED AnyTop (A1v3/A2) and search over z (cross-attention input)
  3. Larger search budget: 64 candidates per pair

Search space for z candidates per pair:
  - Source's own encoded z (the "content latent" of the source clip)
  - 16 random other-source encoded z's (control: should give higher L_eff if z carries content)
  - 16 target's own-skeleton z's (control: target's natural z)
  - 16 null_z + perturbations
  - 15 random Gaussian z's

If true_source z reliably beats controls → z carries cross-skeleton transferable content.

Usage:
    conda run -n anytop python -m eval.z_star_rigor_v2 --n_pairs 24
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
    p.add_argument('--ckpt', default='save/A1v3_infonce_bs_4_latentdim_256/model000199999.pt')
    p.add_argument('--args_json', default='save/A1v3_infonce_bs_4_latentdim_256/args.json')
    p.add_argument('--n_pairs', type=int, default=24)
    p.add_argument('--n_candidates_per_class', type=int, default=16)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--out', type=str, default='eval/results/z_star_rigor_v2.json')
    return p.parse_args()


def load_normalized_psi_loss():
    """Load per-component statistics for normalized L_eff."""
    with open('eval/results/effect_cache/psi_stats.json') as f:
        stats = json.load(f)

    component_slices = {'tau': (0, 8), 'mu': (8, 32), 'eta': (32, 50), 'rho': (50, 62)}
    full_mean = np.array(stats['mean'])
    full_std = np.array(stats['std'])

    def normalized_l_eff(psi_gen, psi_target):
        """Per-component z-scored L1 loss with equal weight per component."""
        gen_norm = (psi_gen - full_mean) / full_std
        tgt_norm = (psi_target - full_mean) / full_std

        comp_losses = {}
        for name, (lo, hi) in component_slices.items():
            comp_losses[name] = float(np.abs(gen_norm[:, lo:hi] - tgt_norm[:, lo:hi]).mean())

        # Equal-weighted total: each component contributes equally regardless of native scale
        comp_losses['total'] = sum(comp_losses[c] for c in component_slices)
        return comp_losses

    return normalized_l_eff


def main():
    args = parse_args()
    from utils.fixseed import fixseed
    from utils import dist_util
    from utils.model_util import create_conditioned_model_and_diffusion, load_model
    from data_loaders.truebones.truebones_utils.get_opt import get_opt
    from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np
    from model.conditioners import T5Conditioner
    from eval.effect_program import extract_effect_program
    from eval.stage1_teacher_audit import build_encode_motion_fn, encoder_z

    fixseed(args.seed)
    dist_util.setup_dist(args.device)
    device = dist_util.dev()

    # Load CONDITIONED AnyTop (A1v3 or A2)
    with open(args.args_json) as f:
        args_d = json.load(f)
    defaults = {'topo_drop_prob': 0.15, 'z_norm_target': None, 'no_rest_pe': False}
    for k, v in defaults.items():
        args_d.setdefault(k, v)
    class NS:
        def __init__(self, d): self.__dict__.update(d)
    m_args = NS(args_d)

    model, diffusion = create_conditioned_model_and_diffusion(m_args)
    state = torch.load(args.ckpt, map_location='cpu')

    # Handle A1v3 z_proj remapping
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state.keys())
    if 'encoder.z_proj.weight' in ckpt_keys and 'encoder.z_proj.1.weight' in model_keys:
        print("  Remapping A1v3 z_proj")
        state['encoder.z_proj.1.weight'] = state.pop('encoder.z_proj.weight')
        state['encoder.z_proj.1.bias'] = state.pop('encoder.z_proj.bias')
        for k in model_keys:
            if k.startswith('encoder.z_proj.0.') or k.startswith('encoder.z_proj.3.'):
                state[k] = model.state_dict()[k]

    load_model(model, state)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"Loaded conditioned AnyTop: {args.ckpt}")

    opt = get_opt(args.device)
    cond_dict = np.load(opt.cond_file, allow_pickle=True).item()
    t5 = T5Conditioner(name=m_args.t5_name, finetune=False, word_dropout=0.0,
                       normalize_text=False, device='cuda')
    n_frames = getattr(m_args, 'num_frames', 120)

    psi_all = np.load('eval/results/effect_cache/psi_all.npy')
    with open('eval/results/effect_cache/clip_metadata.json') as f:
        metadata = json.load(f)
    print(f"Loaded {len(psi_all)} cached ψ vectors")

    with open('dataset/truebones/zoo/truebones_processed/motion_labels.json') as f:
        label_map = json.load(f)

    normalized_l_eff = load_normalized_psi_loss()

    # Generate test pairs
    skels = sorted(set(m['skeleton'] for m in metadata))
    rng = np.random.default_rng(args.seed)
    skel_to_idx = {s: [i for i, m in enumerate(metadata) if m['skeleton'] == s] for s in skels}

    pairs = []
    while len(pairs) < args.n_pairs:
        src_skel, tgt_skel = rng.choice(skels, size=2, replace=False)
        if src_skel == tgt_skel or not skel_to_idx[src_skel] or tgt_skel not in cond_dict:
            continue
        if not skel_to_idx[tgt_skel]:
            continue
        src_idx = int(rng.choice(skel_to_idx[src_skel]))
        pairs.append((src_skel, tgt_skel, src_idx))

    # Lazy-encode only needed clips. First identify which clips we'll need.
    prepare = build_encode_motion_fn(opt, cond_dict, label_map, t5, n_frames, device)

    # Determine which clips need encoding
    needed_indices = set()
    for src_skel, tgt_skel, src_idx in pairs:
        needed_indices.add(src_idx)
        # Other-source pool: clips from skeletons NOT src/tgt
        # Target pool: clips of tgt skeleton
        for i, m in enumerate(metadata):
            if m['skeleton'] == tgt_skel:
                needed_indices.add(i)
    # Plus extra for "other-source" pool — sample 200 random clips
    extra = list(rng.choice(len(metadata), size=min(200, len(metadata)), replace=False))
    needed_indices.update(int(x) for x in extra)
    needed_indices = sorted(needed_indices)

    print(f"\nLazy-encoding z for {len(needed_indices)} clips (needed for {len(pairs)} pairs)...")
    t0 = time.time()
    z_bank = {}
    for k, i in enumerate(needed_indices):
        item = prepare(metadata[i]['fname'])
        if item is None:
            continue
        with torch.no_grad():
            z = encoder_z(model, item)
        z_bank[i] = z.cpu()
        if (k + 1) % 50 == 0:
            print(f"  encoded {k+1}/{len(needed_indices)} ({time.time()-t0:.0f}s)")
    print(f"Latent bank: {len(z_bank)} z vectors ({time.time()-t0:.0f}s)")

    # Build target item dict for sampling
    print(f"\nRunning rigor test on {len(pairs)} pairs...")
    main_results = []

    @torch.no_grad()
    def sample_with_z(target_item, z, max_joints, feature_len, n_frames):
        """Sample motion from conditioned AnyTop with provided z."""
        y = {k: v for k, v in target_item['y'].items()}
        y['z'] = z.to(device)
        x = torch.randn(1, max_joints, feature_len, n_frames, device=device)
        for t_val in reversed(range(diffusion.num_timesteps)):
            t = torch.tensor([t_val], device=device)
            out = diffusion.p_mean_variance(model, x, t, model_kwargs={'y': y}, clip_denoised=False)
            x = out['mean']
            if t_val > 0:
                noise = torch.randn_like(x)
                x = x + torch.exp(0.5 * out['log_variance']) * noise
        return x

    for pair_i, (src_skel, tgt_skel, src_idx) in enumerate(pairs):
        if src_idx not in z_bank:
            continue

        src_z = z_bank[src_idx]
        src_psi = psi_all[src_idx]

        # Build target item once
        tgt_clips = [i for i in skel_to_idx[tgt_skel] if i in z_bank]
        if not tgt_clips:
            continue
        tgt_sample_item = prepare(metadata[tgt_clips[0]]['fname'])
        if tgt_sample_item is None:
            continue

        info = cond_dict[tgt_skel]
        n_joints = len(info['joints_names'])
        parents = info['parents'][:n_joints]
        offsets = info['offsets'][:n_joints]
        mean = info['mean'][:n_joints]
        std = info['std'][:n_joints]

        max_joints = opt.max_joints
        feature_len = opt.feature_len

        # Build candidate z's: 4 categories, then score each
        # 1. SOURCE z (1 candidate — the "true" z to test)
        # 2. Other-source z (random clips from skeletons NOT src_skel and NOT tgt_skel)
        other_src_indices = [i for i in z_bank if metadata[i]['skeleton'] not in {src_skel, tgt_skel}]
        other_z = [z_bank[int(rng.choice(other_src_indices))] for _ in range(args.n_candidates_per_class)]
        # 3. Target's own-skeleton z (control: target's natural z)
        own_z = [z_bank[int(rng.choice(tgt_clips))] for _ in range(args.n_candidates_per_class)]
        # 4. Null z and random z
        null_z = model.null_z.expand(1, src_z.shape[1], src_z.shape[2], src_z.shape[3]).cpu()
        random_z = [torch.randn_like(src_z) * src_z.std() for _ in range(args.n_candidates_per_class - 1)]

        all_candidates = (
            [('source', src_z)] +
            [(f'other_src_{i}', z) for i, z in enumerate(other_z)] +
            [(f'own_skel_{i}', z) for i, z in enumerate(own_z)] +
            [('null', null_z)] +
            [(f'random_{i}', z) for i, z in enumerate(random_z)]
        )

        # Sample motions and score
        scores = []
        t0 = time.time()
        for kind, z in all_candidates:
            sample = sample_with_z(tgt_sample_item, z, max_joints, feature_len, n_frames)
            motion_norm = sample[0].cpu().permute(2, 0, 1).numpy()
            motion_denorm = motion_norm[:, :n_joints] * (std + 1e-6) + mean
            try:
                positions = recover_from_bvh_ric_np(motion_denorm)
                eff = extract_effect_program(positions, parents, offsets)
                psi_gen = eff['psi']
                if np.isnan(psi_gen).any() or np.isinf(psi_gen).any():
                    scores.append((kind, None))
                    continue
                losses = normalized_l_eff(psi_gen, src_psi)
                scores.append((kind, losses))
            except Exception:
                scores.append((kind, None))

        elapsed = time.time() - t0
        valid = [(k, s) for k, s in scores if s is not None]
        if not valid:
            continue

        l_source = next((s['total'] for k, s in valid if k == 'source'), None)
        l_other = [s['total'] for k, s in valid if k.startswith('other_src')]
        l_own = [s['total'] for k, s in valid if k.startswith('own_skel')]
        l_null = next((s['total'] for k, s in valid if k == 'null'), None)
        l_random = [s['total'] for k, s in valid if k.startswith('random')]

        # Source's z rank among ALL candidates (lower rank = better = more source-specific signal)
        all_totals = [s['total'] for _, s in valid]
        source_rank = (np.array(all_totals) < l_source).sum() if l_source is not None else None

        result = {
            'src_skel': src_skel,
            'tgt_skel': tgt_skel,
            'src_idx': src_idx,
            'l_source':           l_source,
            'l_other_src_mean':   float(np.mean(l_other)) if l_other else None,
            'l_own_skel_mean':    float(np.mean(l_own)) if l_own else None,
            'l_null':             l_null,
            'l_random_mean':      float(np.mean(l_random)) if l_random else None,
            'source_rank':        int(source_rank) if source_rank is not None else None,
            'n_candidates':       len(valid),
        }
        main_results.append(result)
        l_src_str = f"{l_source:.3f}" if l_source is not None else "N/A"
        l_other_str = f"{result['l_other_src_mean']:.3f}" if result['l_other_src_mean'] is not None else "N/A"
        if (pair_i + 1) % 4 == 0 or pair_i == 0:
            print(f"  [{pair_i+1}/{len(pairs)}] {src_skel}→{tgt_skel}: "
                  f"L_src={l_src_str} L_other={l_other_str} rank={source_rank} ({elapsed:.0f}s)")

    # Statistical analysis
    print(f"\n{'='*70}")
    print(f"RIGOR TEST V2 — n_pairs={len(main_results)}, normalized loss + conditioned model + larger search")
    print(f"{'='*70}")

    l_src = np.array([r['l_source'] for r in main_results if r['l_source'] is not None])
    l_other = np.array([r['l_other_src_mean'] for r in main_results if r['l_other_src_mean'] is not None])
    l_own = np.array([r['l_own_skel_mean'] for r in main_results if r['l_own_skel_mean'] is not None])
    l_null = np.array([r['l_null'] for r in main_results if r['l_null'] is not None])
    l_random = np.array([r['l_random_mean'] for r in main_results if r['l_random_mean'] is not None])

    print(f"\nL_eff (normalized) by candidate class:")
    print(f"  source (true)     : {l_src.mean():.4f} ± {l_src.std():.4f}")
    print(f"  other-source ctrl : {l_other.mean():.4f} ± {l_other.std():.4f}")
    print(f"  target-own ctrl   : {l_own.mean():.4f} ± {l_own.std():.4f}")
    print(f"  null              : {l_null.mean():.4f} ± {l_null.std():.4f}")
    print(f"  random            : {l_random.mean():.4f} ± {l_random.std():.4f}")

    # Paired tests
    if len(l_src) >= 6 and len(l_other) >= 6:
        wp_other, _ = scipy_stats.wilcoxon(l_src, l_other, alternative='less'), None
        wp_other_p = scipy_stats.wilcoxon(l_src, l_other, alternative='less').pvalue
        cohen_other = (l_src - l_other).mean() / (l_src - l_other).std()
        print(f"\nPaired Wilcoxon (source < other-source):")
        print(f"  p = {wp_other_p:.4f}, Cohen's d = {cohen_other:.3f}")
        if wp_other_p < 0.05:
            print(f"  ✓ SIGNIFICANT — source z provides cross-skeleton signal")
        else:
            print(f"  ✗ NOT SIGNIFICANT — source z does not beat other-source baseline")

    # Source rank distribution
    ranks = [r['source_rank'] for r in main_results if r['source_rank'] is not None]
    print(f"\nSource z rank among all candidates (0 = best):")
    print(f"  mean rank: {np.mean(ranks):.1f} / {main_results[0]['n_candidates']-1 if main_results else 0}")
    print(f"  source in top 5: {sum(r < 5 for r in ranks)}/{len(ranks)}")
    print(f"  source rank #1:  {sum(r == 0 for r in ranks)}/{len(ranks)}")

    out = {
        'n_pairs': len(main_results),
        'l_src_mean': float(l_src.mean()),
        'l_other_mean': float(l_other.mean()),
        'l_own_mean': float(l_own.mean()),
        'l_null_mean': float(l_null.mean()),
        'l_random_mean': float(l_random.mean()),
        'wilcoxon_p_src_vs_other': float(wp_other_p) if 'wp_other_p' in dir() else None,
        'cohen_d_src_vs_other': float(cohen_other) if 'cohen_other' in dir() else None,
        'source_rank_mean': float(np.mean(ranks)),
        'source_in_top5': int(sum(r < 5 for r in ranks)),
        'pairs': main_results,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {args.out}")


if __name__ == '__main__':
    main()
