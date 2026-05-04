"""ACE Stage 2 training loop (paper-faithful per ACE_DESIGN_V3).

Key features:
- D operates on decoded transitions (x_prev, x_cur), trained with R1 zero-centered GP
- G predicts target latent given (z_src, prev_z_tgt) with target-conditioned START token
- Adversarial loss + paper Ψ feature loss (per-frame, body-length-normalized)
- D frozen during G step (prevent grad pollution)
- Stratified pair sampling (50/50 EE-cluster vs uniform)
- Mode-collapse probe bank (3 diagnostics + D-saturation signal)

Usage:
  python -m train.train_ace --scope all --run_name ace_primary_70 --max_steps 50000
  python -m train.train_ace --scope train_v3 --run_name ace_inductive_60 --max_steps 50000
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loaders.truebones.truebones_utils.param_utils import OBJECT_SUBSETS_DICT
from model.ace.generator import ACEGenerator, ACEStartTokens, count_parameters as count_g
from model.ace.discriminator import ACEDiscriminator, J_MAX, count_parameters as count_d
from model.moreflow.skel_graph import (
    SkelGraphEncoder, build_skel_features, pad_to_max_joints,
)
from model.moreflow.stage_a_registry import StageARegistry
from eval.ace_phi import precompute_ace_descriptors
from eval.ace_phi_torch import (
    torch_ace_psi_per_frame, compute_L_feat_torch, gate_ace_or_abort,
)
from eval.moreflow_phi import load_contact_groups

DATA_ROOT = PROJECT_ROOT / 'dataset/truebones/zoo/truebones_processed'
COND_PATH = DATA_ROOT / 'cond.npy'
CACHE_ROOT = PROJECT_ROOT / 'save/moreflow_flow'
SAVE_ROOT = PROJECT_ROOT / 'save/ace'


def lr_lambda(step, warmup, total):
    if warmup > 0 and step < warmup:
        return float(step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_skel_cluster_partition(skel_descs, skels):
    """Group skels by EE count cluster: A(0-2), B(3-4), C(5-6), D(7-8)."""
    clusters = {'A': [], 'B': [], 'C': [], 'D': []}
    for s in skels:
        n = skel_descs[s]['n_ee']
        if n <= 2: clusters['A'].append(s)
        elif n <= 4: clusters['B'].append(s)
        elif n <= 6: clusters['C'].append(s)
        else: clusters['D'].append(s)
    return clusters


def stratified_sample_skel_pair(skels, clusters, rng, p_cluster_uniform=0.5):
    """Sample (src_skel, tgt_skel) with 50% cluster-uniform, 50% pair-uniform."""
    cluster_keys = sorted(clusters.keys())
    if rng.random() < p_cluster_uniform:
        # Cluster-uniform: pick (src_cluster, tgt_cluster) uniformly, then random skel within
        src_c = rng.choice(cluster_keys)
        tgt_c = rng.choice(cluster_keys)
        if not clusters[src_c] or not clusters[tgt_c]:
            return rng.choice(skels), rng.choice(skels)
        return rng.choice(clusters[src_c]), rng.choice(clusters[tgt_c])
    else:
        return rng.choice(skels), rng.choice(skels)


def train(args):
    save_dir = SAVE_ROOT / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / 'training_log.jsonl'

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}, run_name: {args.run_name}, scope: {args.scope}")

    # Load cache
    cache_path = CACHE_ROOT / f'cache_{args.scope}.pt'
    if not cache_path.exists():
        raise FileNotFoundError(f"{cache_path} not found.")
    print(f"Loading cache: {cache_path}")
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    skels = [k for k in cache.keys() if k != '_meta']
    print(f"Cached skels: {len(skels)}")
    # Verify cache is ACE-extended
    if 'prev_row_idx' not in cache[skels[0]]:
        raise RuntimeError(f"Cache lacks ACE extension. Run scripts/moreflow_extend_cache_for_ace.py --scope {args.scope}")

    # Pin per-skel cache to GPU (z_continuous, phi_*, prev_row_idx, is_clip_start)
    print("Pinning cache to GPU...")
    for skel in skels:
        for key, val in cache[skel].items():
            if isinstance(val, torch.Tensor):
                cache[skel][key] = val.to(device)

    # Stage A registry (frozen tokenizers, fp16 to save ~3GB VRAM with 70 tokenizers)
    tokenizer_dtype = torch.float16 if args.tokenizer_fp16 else torch.float32
    print(f"Loading Stage A tokenizer registry (model_dtype={tokenizer_dtype})...")
    registry = StageARegistry(skels, device=device, model_dtype=tokenizer_dtype)
    skel_to_id = {s: i for i, s in enumerate(skels)}
    n_skels = len(skels)
    n_train_skels = n_skels  # for transductive: all skels in scope are "train"; inductive same scope

    # Skel descriptors for ACE Ψ
    print("Building ACE Ψ descriptors...")
    cond_dict = np.load(COND_PATH, allow_pickle=True).item()
    contact_groups = load_contact_groups()
    ace_descs = precompute_ace_descriptors(cond_dict, contact_groups)

    # Pre-compute body_length, ee_joints, n_ee tensors per skel for fast lookup
    body_length_per_skel = torch.zeros(n_skels, device=device)
    ee_joints_per_skel = -torch.ones(n_skels, 8, dtype=torch.long, device=device)  # G_MAX=8
    n_ee_per_skel = torch.zeros(n_skels, dtype=torch.long, device=device)
    for s in skels:
        d = ace_descs[s]
        body_length_per_skel[skel_to_id[s]] = d['body_length']
        ee_joints_per_skel[skel_to_id[s]] = torch.from_numpy(d['ee_joints']).to(device)
        n_ee_per_skel[skel_to_id[s]] = d['n_ee']

    # Skel-graph features
    print("Building skel-graph features...")
    skel_features = build_skel_features(cond_dict)
    max_J = max(skel_features[s]['n_joints'] for s in skels)
    print(f"max_J: {max_J}")
    pj_padded = {}
    pj_mask = {}
    pj_agg = {}
    for s in skels:
        pj, m = pad_to_max_joints(skel_features[s]['per_joint'], max_J)
        pj_padded[s] = pj.to(device)
        pj_mask[s] = m.to(device)
        pj_agg[s] = skel_features[s]['agg'].to(device)

    # Joint-mask per skel for D (using J_MAX=142)
    joint_mask_per_skel = torch.zeros(n_skels, J_MAX, device=device)
    n_joints_per_skel = torch.zeros(n_skels, dtype=torch.long, device=device)
    for s in skels:
        n_j = registry.get(s)['n_joints']
        joint_mask_per_skel[skel_to_id[s], :n_j] = 1.0
        n_joints_per_skel[skel_to_id[s]] = n_j

    # Compute per-skel mean of real z_tgt for START tokens (paper §3.1, target-conditioned)
    print("Computing per-skel z_real means for START tokens...")
    z_means = torch.zeros(n_skels, 8, 256, device=device)
    for s in skels:
        z_means[skel_to_id[s]] = cache[s]['z_continuous'].mean(dim=0)

    # Hard gate
    if not args.skip_gate:
        print("\n=== HARD GATE: numpy ↔ torch ACE Ψ ===")
        try:
            gate_ace_or_abort(cond_dict, contact_groups, n_windows=20, n_skels=5,
                              devices=('cpu',), dtypes=(torch.float64,))
        except RuntimeError as e:
            print(str(e))
            with open(save_dir / 'STATUS_GATE_FAILED.txt', 'w') as f:
                f.write(str(e))
            sys.exit(1)
        print("Gate PASS.\n")

    # Build models
    print("Building Generator + Discriminator + SkelGraphEncoder + StartTokens...")
    G = ACEGenerator(n_skels=n_skels).to(device)
    D = ACEDiscriminator(n_skels=n_skels).to(device)
    skel_enc = SkelGraphEncoder(d_in=6, d_hidden=64, d_agg=4, d_out=128).to(device)
    starts = ACEStartTokens(
        n_skels=n_skels, n_train_skels=n_train_skels,
        codebook_dim=256, n_tokens=8, z_means=z_means,
    ).to(device)
    n_g, n_d, n_s = count_g(G) + count_g(skel_enc) + count_g(starts), count_d(D), count_g(starts)
    print(f"G+enc+starts params: {n_g:,} ({n_g/1e6:.1f}M)")
    print(f"D params:           {n_d:,} ({n_d/1e6:.1f}M)")

    # Optimizers (Adam GAN-standard betas)
    g_params = list(G.parameters()) + list(skel_enc.parameters()) + list(starts.parameters())
    g_optim = torch.optim.Adam(g_params, lr=args.lr_g, betas=(0.5, 0.9))
    d_optim = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.5, 0.9))

    # Cluster partition
    clusters = build_skel_cluster_partition(ace_descs, skels)
    print(f"Skel clusters by EE count: " + ", ".join(f"{k}={len(v)}" for k, v in clusters.items()))

    # ===== Mode-collapse probe bank (50 stratified pairs × 32 fixed source samples) =====
    # 3 diagnostics per checkpoint: latent variance ratio, codebook perplexity,
    # decoded-Ψ pairwise distance variance. Abort if ANY trips for 3 consecutive checks.
    print("Building mode-collapse probe bank...")
    probe_pairs = []
    cluster_keys_nonempty = [k for k, v in clusters.items() if v]
    probe_rng = np.random.RandomState(args.seed + 17)
    if cluster_keys_nonempty:
        # 10 within-cluster pairs (sample from largest cluster)
        largest = max(cluster_keys_nonempty, key=lambda k: len(clusters[k]))
        for _ in range(min(10, len(clusters[largest])**2)):
            probe_pairs.append((probe_rng.choice(clusters[largest]), probe_rng.choice(clusters[largest])))
        # 20 cross-cluster (random pairs from different clusters)
        for _ in range(20):
            ck1, ck2 = probe_rng.choice(cluster_keys_nonempty, 2, replace=True)
            probe_pairs.append((probe_rng.choice(clusters[ck1]), probe_rng.choice(clusters[ck2])))
        # 20 fully random
        for _ in range(20):
            probe_pairs.append((probe_rng.choice(skels), probe_rng.choice(skels)))
    probe_n_samples = 32
    # Pre-sample fixed source windows per probe pair
    probe_data = []
    for src_p, tgt_p in probe_pairs:
        n_src_p = cache[src_p]['z_continuous'].shape[0]
        n_tgt_p = cache[tgt_p]['z_continuous'].shape[0]
        if n_src_p < probe_n_samples or n_tgt_p < probe_n_samples:
            continue
        src_idx_p = torch.from_numpy(probe_rng.choice(n_src_p, probe_n_samples, replace=False)).to(device)
        tgt_idx_p = torch.from_numpy(probe_rng.choice(n_tgt_p, probe_n_samples, replace=False)).to(device)
        # Calibration baseline: real z_tgt variance + decoded-Ψ pairwise variance
        z_real_tgt_p = cache[tgt_p]['z_continuous'][tgt_idx_p]            # [32, 8, 256]
        var_z_real = float(z_real_tgt_p.var().item())
        # Compute decoded-Ψ pairwise variance for real (used as baseline for diagnostic 3)
        with torch.no_grad():
            x_real_norm = registry.decode_tokens(tgt_p, z_real_tgt_p)
            x_real_phys = registry.unnormalize(tgt_p, x_real_norm)
            tgt_p_id = skel_to_id[tgt_p]
            psi_real = torch_ace_psi_per_frame(
                x_real_phys,
                body_length_per_skel[tgt_p_id].expand(probe_n_samples),
                ee_joints_per_skel[tgt_p_id].unsqueeze(0).expand(probe_n_samples, -1),
                n_ee_per_skel[tgt_p_id].expand(probe_n_samples),
            )                                                              # [32, T, 37]
            psi_real_mean = psi_real.mean(dim=1)                           # [32, 37]
            var_psi_real = float(psi_real_mean.var().item())
        probe_data.append({
            'src': src_p, 'tgt': tgt_p,
            'src_idx': src_idx_p,
            'var_z_real_baseline': var_z_real,
            'var_psi_real_baseline': var_psi_real,
        })
    print(f"  {len(probe_data)} probe pairs initialized")
    # Free fragmented GPU memory from probe-bank decode operations
    torch.cuda.empty_cache()

    # Track 3 diagnostics across recent checkpoints
    probe_history = {'latent_var_ratio': [], 'perplexity': [], 'psi_var_ratio': []}

    # Save args
    with open(save_dir / 'args.json', 'w') as f:
        json.dump({**vars(args), 'n_skels': n_skels, 'skels': skels,
                   'max_J': max_J, 'n_g_params': n_g, 'n_d_params': n_d}, f, indent=2)

    # Resume
    rng = np.random.RandomState(args.seed + 1)
    start_step = 0
    ckpts = sorted(save_dir.glob('ckpt_step*.pt'))
    if ckpts and not args.no_resume:
        latest = ckpts[-1]
        print(f"Resuming from {latest.name}")
        sd = torch.load(latest, map_location=device, weights_only=False)
        G.load_state_dict(sd['G'])
        D.load_state_dict(sd['D'])
        skel_enc.load_state_dict(sd['skel_enc'])
        starts.load_state_dict(sd['starts'])
        if args.reset_g_optim:
            print("  --reset_g_optim: G/optim Adam state freshly initialized (G loss scale changed)")
        else:
            g_optim.load_state_dict(sd['g_optim'])
        d_optim.load_state_dict(sd['d_optim'])
        start_step = sd['step']
        rng.set_state(sd['train_rng_state'])

    # Training loop
    G.train()
    D.train()
    skel_enc.train()
    starts.train()
    bce = nn.BCEWithLogitsLoss()
    t0 = time.time()
    gamma_r1 = args.gamma_r1

    # Tracker windows
    d_real_window = deque(maxlen=200)
    d_fake_window = deque(maxlen=200)
    l_adv_window = deque(maxlen=200)
    l_feat_window = deque(maxlen=200)

    # Adaptive batch sizing: D's joint-axis attention is O(J²); for high-J targets
    # (Dragon=142) the math SDP + R1 double-backward can OOM. Scale B by J.
    def adaptive_batch_size(J_tgt, base_B=args.batch_size):
        if J_tgt > 120: return max(2, base_B // 8)         # Dragon=142 → B=4 if base=32, B=1 if base=8
        if J_tgt > 90:  return max(4, base_B // 4)
        if J_tgt > 60:  return max(8, base_B // 2)
        return base_B

    for step in range(start_step, args.max_steps):
        # 1. Sample (src_skel, tgt_skel) stratified
        src_skel, tgt_skel = stratified_sample_skel_pair(skels, clusters, rng,
                                                          p_cluster_uniform=args.p_cluster_uniform)
        src_id_int = skel_to_id[src_skel]
        tgt_id_int = skel_to_id[tgt_skel]

        # Adapt batch size to target's joint count (avoid OOM on Dragon-class targets)
        eff_B = adaptive_batch_size(int(n_joints_per_skel[tgt_id_int].item()))

        # 2. Sample microbatch B from each cache
        N_src = cache[src_skel]['z_continuous'].shape[0]
        N_tgt = cache[tgt_skel]['z_continuous'].shape[0]
        if N_tgt < 4:
            continue
        # Sample tgt windows from FULL pool (include clip-start so START tokens get gradient)
        src_idx = torch.from_numpy(rng.choice(N_src, eff_B)).to(device)
        tgt_idx = torch.from_numpy(rng.choice(N_tgt, eff_B)).to(device)

        prev_row_idx_tgt = cache[tgt_skel]['prev_row_idx'][tgt_idx]       # [B] (-1 = clip start)
        is_start = (prev_row_idx_tgt < 0)
        # Pull z_src, z_cur_tgt, z_prev_tgt
        z_src = cache[src_skel]['z_continuous'][src_idx]                  # [B, 8, 256]
        z_cur_tgt = cache[tgt_skel]['z_continuous'][tgt_idx]              # [B, 8, 256]
        prev_idx_safe = prev_row_idx_tgt.clamp(min=0)
        z_prev_tgt_real = cache[tgt_skel]['z_continuous'][prev_idx_safe]  # [B, 8, 256]
        # Compute START vectors (target-conditioned). starts is in g_params → gets gradient.
        tgt_id_for_start = torch.full((eff_B,), tgt_id_int, dtype=torch.long, device=device)
        start_vec = starts(tgt_id_for_start)                              # [B, 8, 256]
        # Final z_prev_tgt: use cached z when valid; START for clip-start (G/L_feat sees both)
        z_prev_tgt = torch.where(is_start.unsqueeze(-1).unsqueeze(-1),
                                 start_vec, z_prev_tgt_real)
        # For D step, we need non-clip-start (since D needs decoded REAL prev motion).
        # Build a mask + downselect for D-step.
        d_valid_mask = ~is_start
        n_d_valid = int(d_valid_mask.sum().item())

        # Skel/graph conditioning (per-iter shared)
        src_pj = pj_padded[src_skel].unsqueeze(0)
        src_m = pj_mask[src_skel].unsqueeze(0)
        src_a = pj_agg[src_skel].unsqueeze(0)
        tgt_pj = pj_padded[tgt_skel].unsqueeze(0)
        tgt_m = pj_mask[tgt_skel].unsqueeze(0)
        tgt_a = pj_agg[tgt_skel].unsqueeze(0)

        src_id_t = torch.full((eff_B,), src_id_int, dtype=torch.long, device=device)
        tgt_id_t = torch.full((eff_B,), tgt_id_int, dtype=torch.long, device=device)
        tgt_jmask_b = joint_mask_per_skel[tgt_id_int].unsqueeze(0).expand(eff_B, -1)
        tgt_graph_for_d = None  # filled in below

        # ==== D STEP ====
        # Per ACE_DESIGN_V3.md §lines 138, 340: D-step excludes clip-start windows entirely.
        # We don't want D to learn that decoded(learnable START) is "real motion".
        # Skip D step entirely if no valid (non-clip-start) rows in microbatch.
        skip_d_step = (n_d_valid == 0)
        if not skip_d_step:
            with torch.no_grad():
                # Decode ONLY valid rows (real z_prev for non-clip-start). Filter first.
                z_prev_real_filt = z_prev_tgt_real[d_valid_mask]                # [n_valid, 8, 256]
                z_cur_real_filt = z_cur_tgt[d_valid_mask]
                x_prev_decoded_norm = registry.decode_tokens(tgt_skel, z_prev_real_filt)
                x_prev_real = registry.unnormalize(tgt_skel, x_prev_decoded_norm)
                x_cur_real_norm = registry.decode_tokens(tgt_skel, z_cur_real_filt)
                x_cur_real = registry.unnormalize(tgt_skel, x_cur_real_norm)

            # Make real input require grad for R1 GP
            x_prev_real_grad = x_prev_real.detach().requires_grad_(True)
            x_cur_real_grad = x_cur_real.detach().requires_grad_(True)

            # Filter conditioning to valid rows only
            tgt_id_t_d = tgt_id_t[d_valid_mask]
            tgt_jmask_b_d = tgt_jmask_b[d_valid_mask]
            src_id_t_d = src_id_t[d_valid_mask]

            # Skel-enc forward (re-run per microbatch for autograd graph isolation)
            src_g_emb_d = skel_enc(src_pj, src_m, src_a).expand(n_d_valid, -1)
            tgt_g_emb_d = skel_enc(tgt_pj, tgt_m, tgt_a).expand(n_d_valid, -1)

            # R1 GP requires double-backward; flash/mem-efficient SDP doesn't support it.
            # Wrap the ENTIRE D-step (forward + backward) in math-mode SDP — backward through gp
            # invokes another grad pass through D's attention, which also needs math-mode.
            with torch.backends.cuda.sdp_kernel(enable_flash=False,
                                                 enable_mem_efficient=False,
                                                 enable_math=True):
                # Detach D's condition input — D-step shouldn't update skel_enc.
                d_real_logit = D(x_prev_real_grad, x_cur_real_grad,
                                 tgt_id_t_d, tgt_g_emb_d.detach(), tgt_jmask_b_d)
                L_D_real = bce(d_real_logit, torch.ones_like(d_real_logit))

                # R1 zero-centered GP on real (both prev and cur)
                gp_grads = torch.autograd.grad(
                    d_real_logit.sum(), [x_prev_real_grad, x_cur_real_grad],
                    create_graph=True, retain_graph=True)
                gp_norm_sq = (gp_grads[0]**2).flatten(1).sum(dim=-1) + (gp_grads[1]**2).flatten(1).sum(dim=-1)
                gp = 0.5 * gamma_r1 * gp_norm_sq.mean()

                # Fake: G forward over VALID rows. Paper-faithful: ACE has no VQ quantization
                # at any stage (z is continuous throughout training and inference).
                # Stage A decoder handles continuous z robustly (already used this way for src
                # motion via cache.z_continuous).
                z_src_d = z_src[d_valid_mask]
                z_prev_d = z_prev_tgt[d_valid_mask]   # all real-prev (mask excluded clip-start)
                with torch.no_grad():
                    z_pred_fake = G(z_src_d, z_prev_d, src_id_t_d, tgt_id_t_d,
                                    src_g_emb_d, tgt_g_emb_d)
                    x_cur_fake_norm = registry.decode_tokens(tgt_skel, z_pred_fake)
                    x_cur_fake = registry.unnormalize(tgt_skel, x_cur_fake_norm)

                # Detach condition for fake forward in D step too
                d_fake_logit = D(x_prev_real, x_cur_fake, tgt_id_t_d,
                                 tgt_g_emb_d.detach(), tgt_jmask_b_d)
                L_D_fake = bce(d_fake_logit, torch.zeros_like(d_fake_logit))
                L_D = L_D_real + L_D_fake + gp
                d_optim.zero_grad()
                L_D.backward()
                torch.nn.utils.clip_grad_norm_(D.parameters(), 1.0)
                d_optim.step()

            d_real_window.append(torch.sigmoid(d_real_logit).mean().item())
            d_fake_window.append(torch.sigmoid(d_fake_logit).mean().item())
        else:
            # No valid D samples; preserve last D logit stats (window holds running avg)
            L_D_real = torch.tensor(0.0, device=device)
            L_D_fake = torch.tensor(0.0, device=device)
            gp = torch.tensor(0.0, device=device)

        # ==== G STEP ====
        # Freeze D's params so D's weights don't get gradient from G.backward.
        # Don't call D.eval() — LayerNorm has no running stats; it can break autograd
        # graph propagation through frozen params combined with eval mode.
        for p in D.parameters(): p.requires_grad_(False)
        try:
            src_g_emb_g = skel_enc(src_pj, src_m, src_a).expand(eff_B, -1)
            tgt_g_emb_g = skel_enc(tgt_pj, tgt_m, tgt_a).expand(eff_B, -1)

            # Decode FULL-batch x_prev (includes decoded START for clip-start rows) for G step.
            # Per design line 340: G step keeps clip-start rows so START tokens get gradient.
            with torch.no_grad():
                x_prev_g_norm = registry.decode_tokens(tgt_skel, z_prev_tgt)
                x_prev_g = registry.unnormalize(tgt_skel, x_prev_g_norm)

            # Paper-faithful: continuous z directly through Stage A decoder (no STE quantize).
            # ACE paper has continuous embedding z ∈ R^32 throughout training; π decodes z directly.
            z_pred = G(z_src, z_prev_tgt, src_id_t, tgt_id_t, src_g_emb_g, tgt_g_emb_g)
            x_cur_fake_norm_g = registry.decode_tokens(tgt_skel, z_pred)
            x_cur_fake_g = registry.unnormalize(tgt_skel, x_cur_fake_norm_g)

            # Adversarial loss — DETACH tgt_g_emb_g passed to D
            d_fake_g_logit = D(x_prev_g, x_cur_fake_g, tgt_id_t, tgt_g_emb_g.detach(), tgt_jmask_b)
            L_adv = bce(d_fake_g_logit, torch.ones_like(d_fake_g_logit))

            # Feature loss (per-frame Ψ)
            x_src_norm = registry.decode_tokens(src_skel, z_src)
            x_src = registry.unnormalize(src_skel, x_src_norm)

            body_src = body_length_per_skel[src_id_int].expand(eff_B)
            body_tgt = body_length_per_skel[tgt_id_int].expand(eff_B)
            ee_src = ee_joints_per_skel[src_id_int].unsqueeze(0).expand(eff_B, -1)
            ee_tgt = ee_joints_per_skel[tgt_id_int].unsqueeze(0).expand(eff_B, -1)
            nee_src = n_ee_per_skel[src_id_int].expand(eff_B)
            nee_tgt = n_ee_per_skel[tgt_id_int].expand(eff_B)

            L_feat = compute_L_feat_torch(x_src, x_cur_fake_g, body_src, body_tgt,
                                           ee_src, ee_tgt, nee_src, nee_tgt)

            L_G = args.w_adv * L_adv + args.w_feat * L_feat

            g_optim.zero_grad()
            L_G.backward()
            torch.nn.utils.clip_grad_norm_(g_params, 1.0)
            g_optim.step()
        finally:
            # Always restore D's trainability — exception-safe (must-fix from local code review)
            for p in D.parameters(): p.requires_grad_(True)
            d_optim.zero_grad()  # safety

        l_adv_window.append(L_adv.item())
        l_feat_window.append(L_feat.item())

        # Logging
        if (step + 1) % args.log_interval == 0:
            elapsed = time.time() - t0
            eta = elapsed / max(1, step + 1 - start_step) * (args.max_steps - step - 1)
            d_real_avg = np.mean(d_real_window)
            d_fake_avg = np.mean(d_fake_window)
            l_adv_avg = np.mean(l_adv_window)
            l_feat_avg = np.mean(l_feat_window)
            print(f"  [{step+1:6d}/{args.max_steps}] L_D_real={L_D_real.item():.3f} "
                  f"L_D_fake={L_D_fake.item():.3f} gp={gp.item():.2e} "
                  f"L_adv={L_adv.item():.3f} L_feat={L_feat.item():.3f} "
                  f"σD_r={d_real_avg:.3f} σD_f={d_fake_avg:.3f} "
                  f"pair={src_skel}→{tgt_skel} eff_B={eff_B} "
                  f"({elapsed:.0f}s, ETA {eta/60:.0f}min)")
            with open(log_path, 'a') as f:
                f.write(json.dumps({
                    'step': step + 1,
                    'L_D_real': L_D_real.item(), 'L_D_fake': L_D_fake.item(), 'gp': gp.item(),
                    'L_adv': L_adv.item(), 'L_feat': L_feat.item(),
                    'sigma_D_real_avg': d_real_avg, 'sigma_D_fake_avg': d_fake_avg,
                    'L_adv_avg200': l_adv_avg, 'L_feat_avg200': l_feat_avg,
                    'src': src_skel, 'tgt': tgt_skel,
                }) + '\n')

        # Periodic checkpoint
        if (step + 1) % args.ckpt_interval == 0 and (step + 1) < args.max_steps:
            ck = save_dir / f'ckpt_step{step+1:07d}.pt'
            tmp = ck.with_suffix('.pt.tmp')
            torch.save({
                'step': step + 1,
                'G': G.state_dict(),
                'D': D.state_dict(),
                'skel_enc': skel_enc.state_dict(),
                'starts': starts.state_dict(),
                'g_optim': g_optim.state_dict(),
                'd_optim': d_optim.state_dict(),
                'train_rng_state': rng.get_state(),
                'skels': skels,
                'skel_to_id': skel_to_id,
            }, tmp)
            os.replace(tmp, ck)
            for old in sorted(save_dir.glob('ckpt_step*.pt'))[:-2]:
                old.unlink()

        # ===== Diagnostic probes every 1K steps (after warmup) =====
        if (step + 1) % 1000 == 0 and (step + 1) >= args.fallback_check_step:
            # 1. D-saturation signal
            d_real_avg = np.mean(d_real_window)
            d_fake_avg = np.mean(d_fake_window)
            d_sat = (d_real_avg > 0.95 and d_fake_avg < 0.05)
            if d_sat:
                with open(save_dir / 'STATUS_D_SATURATED.txt', 'w') as f:
                    f.write(f"step={step+1} σD_real={d_real_avg:.3f} σD_fake={d_fake_avg:.3f}")

            # 2. Mode-collapse probe bank (3 diagnostics)
            G.eval()
            starts.eval()
            with torch.no_grad():
                ratios_lat, perps, ratios_psi = [], [], []
                for probe in probe_data:
                    src_p, tgt_p = probe['src'], probe['tgt']
                    src_p_id = skel_to_id[src_p]
                    tgt_p_id = skel_to_id[tgt_p]
                    z_src_p = cache[src_p]['z_continuous'][probe['src_idx']]   # [32, 8, 256]
                    # START as prev (probe simulates inference's first chunk)
                    tgt_id_t_p = torch.full((probe_n_samples,), tgt_p_id, dtype=torch.long, device=device)
                    z_prev_p = starts(tgt_id_t_p)                              # [32, 8, 256]
                    # Skel/graph features
                    src_emb_p = skel_enc(pj_padded[src_p].unsqueeze(0), pj_mask[src_p].unsqueeze(0),
                                          pj_agg[src_p].unsqueeze(0)).expand(probe_n_samples, -1)
                    tgt_emb_p = skel_enc(pj_padded[tgt_p].unsqueeze(0), pj_mask[tgt_p].unsqueeze(0),
                                          pj_agg[tgt_p].unsqueeze(0)).expand(probe_n_samples, -1)
                    src_id_t_p = torch.full((probe_n_samples,), src_p_id, dtype=torch.long, device=device)
                    z_pred_p = G(z_src_p, z_prev_p, src_id_t_p, tgt_id_t_p, src_emb_p, tgt_emb_p)

                    # Diagnostic 1: latent variance ratio
                    var_pred = float(z_pred_p.var().item())
                    ratios_lat.append(var_pred / max(probe['var_z_real_baseline'], 1e-9))

                    # Diagnostic 2: codebook perplexity
                    cb = registry.codebook(tgt_p, padded_to=0).float()    # cast for fp16 tokenizer compat
                    dists = torch.cdist(z_pred_p.reshape(-1, 256).unsqueeze(0),
                                         cb.unsqueeze(0)).squeeze(0)
                    indices = dists.argmin(dim=-1)                              # [32*8]
                    K_eff = registry.get(tgt_p)['K_eff']
                    counts = torch.bincount(indices, minlength=K_eff).float()
                    p_dist = counts / counts.sum()
                    entropy = -(p_dist * (p_dist + 1e-10).log()).sum()
                    perps.append(float((entropy / math.log(K_eff)).item()))    # normalized entropy

                    # Diagnostic 3: decoded-Ψ pairwise distance variance ratio
                    z_pred_p_q = registry.ste_quantize(tgt_p, z_pred_p)
                    x_pred_norm = registry.decode_tokens(tgt_p, z_pred_p_q)
                    x_pred_phys = registry.unnormalize(tgt_p, x_pred_norm)
                    psi_pred = torch_ace_psi_per_frame(
                        x_pred_phys,
                        body_length_per_skel[tgt_p_id].expand(probe_n_samples),
                        ee_joints_per_skel[tgt_p_id].unsqueeze(0).expand(probe_n_samples, -1),
                        n_ee_per_skel[tgt_p_id].expand(probe_n_samples),
                    )
                    psi_pred_mean = psi_pred.mean(dim=1)
                    var_psi_pred = float(psi_pred_mean.var().item())
                    ratios_psi.append(var_psi_pred / max(probe['var_psi_real_baseline'], 1e-9))
            G.train()
            starts.train()

            mean_ratio_lat = float(np.mean(ratios_lat))
            mean_perp = float(np.mean(perps))
            mean_ratio_psi = float(np.mean(ratios_psi))
            probe_history['latent_var_ratio'].append(mean_ratio_lat)
            probe_history['perplexity'].append(mean_perp)
            probe_history['psi_var_ratio'].append(mean_ratio_psi)
            print(f"  PROBE [{step+1}]: lat_var_ratio={mean_ratio_lat:.3f} "
                  f"perp_norm={mean_perp:.3f} psi_var_ratio={mean_ratio_psi:.3f} "
                  f"σD={d_real_avg:.2f}/{d_fake_avg:.2f}")
            with open(log_path, 'a') as f:
                f.write(json.dumps({
                    'step': step + 1, 'event': 'probe',
                    'lat_var_ratio': mean_ratio_lat,
                    'perp_norm': mean_perp,
                    'psi_var_ratio': mean_ratio_psi,
                    'd_sat': bool(d_sat),
                }) + '\n')

            # Abort criterion: ANY signal triggers if declining for 3 consecutive checkpoints
            def _trip(history, key, thresh):
                hist = probe_history[key][-3:]
                if len(hist) < 3:
                    return False
                # Declining + below threshold for last 3
                return all(h < thresh for h in hist) and hist[-1] < hist[0]

            # Explosion detector: latent variance ratio > 1e6 OR non-finite means G output is unbounded
            # (typical decoded-to-codebook collapse where all outputs map to single entry).
            if not np.isfinite(mean_ratio_lat) or mean_ratio_lat > 1e6:
                msg = f"Latent variance EXPLOSION at step {step+1}: ratio={mean_ratio_lat:.2e}"
                print(f"  GENERATOR EXPLOSION: {msg}")
                with open(save_dir / 'STATUS_G_EXPLODED.txt', 'w') as f:
                    f.write(msg)
                sys.exit(2)
            if _trip(probe_history, 'latent_var_ratio', 0.1):
                msg = f"Latent variance ratio collapse at step {step+1}: {probe_history['latent_var_ratio'][-3:]}"
                print(f"  MODE COLLAPSE TRIGGERED: {msg}")
                with open(save_dir / 'STATUS_MODE_COLLAPSE.txt', 'w') as f:
                    f.write(msg)
                sys.exit(2)
            # Codebook perplexity abort DISABLED for paper-faithful run.
            # Perplexity measures codebook usage when we WOULD-quantize G's output. Since the
            # paper-faithful pipeline uses continuous decoding throughout (no quantization at
            # train or inference per ACE paper), perp_norm is purely diagnostic — not a valid
            # collapse signal. The true mode-collapse signal is psi_var_ratio (decoded motion
            # diversity), which remains a hard abort criterion below.
            if _trip(probe_history, 'psi_var_ratio', 0.1):
                msg = f"Decoded-Ψ variance collapse at step {step+1}: {probe_history['psi_var_ratio'][-3:]}"
                print(f"  MODE COLLAPSE TRIGGERED: {msg}")
                with open(save_dir / 'STATUS_MODE_COLLAPSE.txt', 'w') as f:
                    f.write(msg)
                sys.exit(2)

    # Final ckpt
    ckpt_path = save_dir / 'ckpt_final.pt'
    tmp = ckpt_path.with_suffix('.pt.tmp')
    torch.save({
        'step': args.max_steps,
        'G': G.state_dict(),
        'D': D.state_dict(),
        'skel_enc': skel_enc.state_dict(),
        'starts': starts.state_dict(),
        'args': vars(args),
        'skels': skels,
        'skel_to_id': skel_to_id,
    }, tmp)
    os.replace(tmp, ckpt_path)
    for stale in save_dir.glob('ckpt_step*.pt'):
        stale.unlink()
    print(f"\nSaved final ckpt: {ckpt_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scope', choices=['train_v3', 'all'], default='all')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--max_steps', type=int, default=50000)
    parser.add_argument('--lr_g', type=float, default=1e-4)
    parser.add_argument('--lr_d', type=float, default=1e-4)
    parser.add_argument('--w_adv', type=float, default=1.0)
    parser.add_argument('--w_feat', type=float, default=1.0)
    parser.add_argument('--gamma_r1', type=float, default=0.1, help='R1 GP weight (paper: 0.1)')
    parser.add_argument('--p_cluster_uniform', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_interval', type=int, default=200)
    parser.add_argument('--ckpt_interval', type=int, default=5000)
    parser.add_argument('--fallback_check_step', type=int, default=10000)
    parser.add_argument('--no_resume', action='store_true')
    parser.add_argument('--reset_g_optim', action='store_true',
                        help='Re-init G optimizer Adam state on resume (use when changing w_feat/w_adv scale)')
    parser.add_argument('--skip_gate', action='store_true')
    parser.add_argument('--tokenizer_fp16', action='store_true', default=True,
                        help='Cast frozen Stage A tokenizers to fp16 (saves ~3 GB VRAM with 70 tokenizers)')
    args = parser.parse_args()

    if args.run_name is None:
        args.run_name = 'ace_primary_70' if args.scope == 'all' else 'ace_inductive_60'

    train(args)


if __name__ == '__main__':
    main()
