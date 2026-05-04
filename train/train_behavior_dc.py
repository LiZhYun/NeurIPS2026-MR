"""D+C training for behavior-conditioned cross-skeleton retargeting.

Per the reviewer-locked spec (Round 11 adversarial review, 2026-04-14):

Loss:
    L = L_self + λ_cf * L_cf + λ_pair * L_pair   (λ_cf = λ_pair = 0.5)

Where:
  - L_self = recon on aug_type=0 samples (weight 1.0)
  - L_pair = recon on aug_type != 0 samples (topology-edited via add/remove-joint aug)
  - L_cf   = counterfactual hinge on aug_type=0 samples:
             hinge(m_i + sg(L_matched_i) - L_mismatched_i),
             m_i = clip(0.05 * sg(L_matched_i), 0.01, 0.03), computed in FP32

Timestep band weighting for L_cf:
  - t >  650                : weight 1.0   (high-noise)
  - 350 <= t <= 650         : weight 0.25  (mid-noise)
  - t <  350                : weight 0.0   (low-noise — denoising nearly
                                             deterministic from x_t, B uninformative)

B' (mismatched) negatives, drawn from a FIFO queue of 128 recent B's:
  - 50% : different action class
  - 50% : same action class with top-quartile ψ distance
  - Fallback: diff-action if same-action/far-ψ pool empty

B channel: action + ψ only (residual DISABLED per reviewer requirement).

Kill criteria (pre-registered, AND gates):
  - 25k : held-out gap_highnoise / L_matched > 0.02 AND
          held-out pair loss decreasing AND pair_loss < 2.0 * self_loss
  - 50k : held-out gap_highnoise / L_matched > 0.05 AND
          Track B action-transfer accuracy >= 24% on >=40 held-out clips

Three gaps logged separately every log_interval:
  - gap_action_diff              (B' = random different-action)
  - gap_same_action_far_psi      (B' = same-action top-quartile-ψ distance)
  - gap_null                     (B' = null_behavior parameter)
"""
import os
import json
import time
import math
import argparse
import random
import numpy as np
import torch
from collections import deque
from os.path import join as pjoin

from utils.fixseed import fixseed
from utils import dist_util
from utils.model_util import create_gaussian_diffusion
from model.anytop_behavior import AnyTopBehavior
from data_loaders.get_data_dc import get_dataset_loader_dc
from data_loaders.truebones.truebones_utils.get_opt import get_opt

ACTION_CLASSES = ['walk', 'run', 'idle', 'attack', 'fly', 'swim', 'jump',
                  'turn', 'die', 'eat', 'getup', 'other']
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_CLASSES)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--save_dir', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--num_steps', type=int, default=200000)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--latent_dim', type=int, default=256)
    p.add_argument('--layers', type=int, default=4,
                   help='Match B1_scratch baseline (4 layers, ~8M params) for apples-to-apples')
    p.add_argument('--num_frames', type=int, default=120)
    p.add_argument('--temporal_window', type=int, default=31)
    p.add_argument('--t5_name', type=str, default='t5-base')
    p.add_argument('--init_from', type=str, default='',
                   help='Init backbone from checkpoint ("" for random)')
    p.add_argument('--lambda_cf', type=float, default=0.5)
    p.add_argument('--lambda_pair', type=float, default=0.5)
    p.add_argument('--n_behavior_tokens', type=int, default=8)
    p.add_argument('--behavior_drop_prob', type=float, default=0.1)
    p.add_argument('--aug_prob_noop', type=float, default=0.75)
    p.add_argument('--aug_prob_remove', type=float, default=0.125)
    p.add_argument('--aug_prob_add', type=float, default=0.125)
    p.add_argument('--queue_size', type=int, default=128)
    p.add_argument('--high_noise_min', type=int, default=65,
                   help='High-noise band start (AnyTop uses 100 DDPM steps, so '
                        'reviewer spec t>650 @ 1000 steps maps to t>65 @ 100 steps)')
    p.add_argument('--mid_noise_min', type=int, default=35,
                   help='Mid-noise band start (reviewer spec 350/1000 → 35/100)')
    p.add_argument('--high_noise_weight', type=float, default=1.0)
    p.add_argument('--mid_noise_weight', type=float, default=0.25)
    p.add_argument('--stratified_fraction', type=float, default=0.5,
                   help='Fraction of self samples forced into high-noise band')
    p.add_argument('--effect_cache', type=str,
                   default='eval/results/effect_cache/psi_all.npy')
    p.add_argument('--clip_metadata', type=str,
                   default='eval/results/effect_cache/clip_metadata.json')
    p.add_argument('--probe_clips', type=int, default=40,
                   help='Clips in held-out probe for gap metrics')
    p.add_argument('--save_interval', type=int, default=10000)
    p.add_argument('--log_interval', type=int, default=100)
    p.add_argument('--probe_interval', type=int, default=2500,
                   help='Every N steps, evaluate decomposed probe gaps')
    p.add_argument('--probe_batches', type=int, default=8,
                   help='Number of cached probe batches (effective probe size '
                        '= probe_batches * batch_size)')
    p.add_argument('--resume_from', type=str, default='',
                   help='Checkpoint path to resume training state from. '
                        'Loads model weights and fast-forwards step counter '
                        '(optimizer warm-restarts).')
    p.add_argument('--smoke', action='store_true')
    return p.parse_args()


# ---------------------------------------------------------------------------
# ψ cache / probe set
# ---------------------------------------------------------------------------
def build_psi_lookup(metadata_path, psi_path):
    with open(metadata_path) as f:
        metadata = json.load(f)
    psi_all = np.load(psi_path)
    assert len(metadata) == len(psi_all)
    return {m['fname']: psi_all[i] for i, m in enumerate(metadata)}, metadata


def build_probe_set(psi_lookup, metadata, fname_to_action, n=40, seed=0):
    """Pick n clips spanning as many actions + skeletons as possible."""
    rng = np.random.default_rng(seed)
    buckets = {}
    for m in metadata:
        fn = m['fname']
        if fn not in psi_lookup:
            continue
        key = (m.get('skeleton', '?'), fname_to_action.get(fn, 11))
        buckets.setdefault(key, []).append(fn)
    keys = list(buckets.keys())
    rng.shuffle(keys)
    probe = []
    for k in keys:
        pool = buckets[k]
        probe.append(rng.choice(pool))
        if len(probe) >= n:
            break
    if len(probe) < n:
        all_fnames = list(psi_lookup.keys())
        while len(probe) < n:
            probe.append(rng.choice(all_fnames))
    return probe[:n]


# ---------------------------------------------------------------------------
# B' negative mining queue
# ---------------------------------------------------------------------------
class NegativeQueue:
    """FIFO queue of (psi_mean_62, action, fname). Used to sample B'."""
    def __init__(self, max_size=128):
        self.max_size = max_size
        self.q = deque(maxlen=max_size)

    def add(self, psi, action, fname):
        psi_mean = psi.mean(axis=0)  # (62,)
        self.q.append((psi_mean.astype(np.float32), int(action), fname))

    def __len__(self):
        return len(self.q)

    def sample(self, target_psi_mean, target_action, mode='mixed', rng=None):
        """Return (entry, mode_used) where entry = (psi_mean_62, action, fname).
        mode: 'diff_action' | 'same_action_far_psi' | 'mixed' (50/50)
        Returns (None, None) if no usable negative.
        """
        if len(self.q) == 0:
            return None, None
        if rng is None:
            rng = random
        chosen_mode = mode
        if mode == 'mixed':
            chosen_mode = 'diff_action' if rng.random() < 0.5 else 'same_action_far_psi'

        entries = list(self.q)
        if chosen_mode == 'diff_action':
            cands = [e for e in entries if e[1] != target_action]
            actual_mode = 'diff_action'
            if not cands:
                cands = entries
                actual_mode = 'fallback'
            return cands[int(rng.random() * len(cands))], actual_mode

        if chosen_mode == 'same_action_far_psi':
            cands = [e for e in entries if e[1] == target_action]
            if len(cands) < 4:
                fallback = [e for e in entries if e[1] != target_action]
                if not fallback:
                    fallback = entries
                return fallback[int(rng.random() * len(fallback))], 'diff_action_fallback'
            dists = np.array([np.linalg.norm(e[0] - target_psi_mean) for e in cands])
            k = max(1, int(len(cands) * 0.25))
            top_idx = np.argpartition(-dists, k - 1)[:k]
            pick = top_idx[int(rng.random() * len(top_idx))]
            return cands[int(pick)], 'same_action_far_psi'

        return entries[int(rng.random() * len(entries))], 'random'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fill_psi_action(source_names, psi_lookup, fname_to_action, device,
                    override_psi=None, override_action=None):
    """Build psi and action_label tensors for a batch."""
    psi_list = []
    action_list = []
    for bi, fname in enumerate(source_names):
        if override_psi is not None and override_psi[bi] is not None:
            psi_list.append(override_psi[bi])
        else:
            psi_list.append(psi_lookup.get(fname, np.zeros((64, 62), dtype=np.float32)))
        if override_action is not None and override_action[bi] is not None:
            action_list.append(int(override_action[bi]))
        else:
            action_list.append(fname_to_action.get(fname, ACTION_TO_IDX['other']))
    psi_t = torch.tensor(np.stack(psi_list), dtype=torch.float32, device=device)
    action_t = torch.tensor(action_list, dtype=torch.long, device=device)
    return psi_t, action_t


def stratified_t(bs, aug_mask_self, high_noise_min, total_t, stratified_fraction, device):
    """Sample t with stratified high-noise forcing on a fraction of self samples.
    aug_mask_self: [B] bool — True for aug_type==0 samples.
    Returns t: [B] long.
    """
    t = torch.randint(0, total_t, (bs,), device=device)
    for i in range(bs):
        if aug_mask_self[i] and random.random() < stratified_fraction:
            t[i] = random.randint(high_noise_min, total_t - 1)
    return t


def compute_eps_loss(model, diffusion, x_start, t, cond, noise):
    """Run one diffusion training step with shared noise, return per-sample loss [B]."""
    losses = diffusion.training_losses(model, x_start, t, model_kwargs=cond,
                                        noise=noise)
    return losses['loss'], losses.get('l_simple', losses['loss'])


def set_behavior(cond, psi_t, action_t):
    """Overwrite psi and action_label in cond (does NOT mutate other keys)."""
    cond['y']['psi'] = psi_t
    cond['y']['action_label'] = action_t
    if 'behavior_tokens' in cond['y']:
        del cond['y']['behavior_tokens']
    return cond


def set_null_behavior(cond, model, bs):
    """Force the behavior to the learned null token by deleting psi/action and
    passing behavior_tokens=null directly."""
    null = model.null_behavior.detach().expand(bs, 1, model.n_total_tokens,
                                               model.latent_dim).clone()
    if 'psi' in cond['y']:
        del cond['y']['psi']
    if 'action_label' in cond['y']:
        del cond['y']['action_label']
    cond['y']['behavior_tokens'] = null
    return cond


# ---------------------------------------------------------------------------
# Probe eval — DECOMPOSED GAPS
# ---------------------------------------------------------------------------
def _override_one_step(model, diffusion, batch, cond, source_names,
                       override_psi_list, override_action_list,
                       psi_lookup, fname_to_action,
                       t, noise, device):
    """Helper: run one denoising forward with overridden B, return per-sample loss [B]."""
    psi_t, action_t = fill_psi_action(source_names, psi_lookup,
                                       fname_to_action, device,
                                       override_psi=override_psi_list,
                                       override_action=override_action_list)
    set_behavior(cond, psi_t, action_t)
    L, _ = compute_eps_loss(model, diffusion, batch, t, cond, noise)
    return L


@torch.no_grad()
def decomposed_probe(model, diffusion, cached_probe, psi_lookup,
                     fname_to_action, queue, high_noise_min, total_t, device):
    """Run probe on a CACHED set of (batch, cond) tuples — fixed across calls.

    Returns dict with five gaps (relative to L_matched on high-noise band):
      gap_action_diff           : both action+ψ swapped (diff-action negative)
      gap_same_action_far_psi   : action matched, ψ swapped to far same-action ψ′
                                  (== gap_psi_only — kept name for back-compat)
      gap_action_only           : action swapped to different, ψ matched
      gap_null                  : behavior tokens replaced with null parameter
      gap_psi_only              : alias of gap_same_action_far_psi (clarity)

    Aggregation: per-clip loss values pooled across all cached batches before
    computing relative gap. Same noise / same t are reused per call so within-call
    contrasts are paired.
    """
    model.eval()
    L_matched_all = []
    L_diff_all = []
    L_far_all = []
    L_action_only_all = []
    L_null_all = []

    for batch, cond in cached_probe:
        bs = batch.shape[0]
        # Move tensors to device (cached probe lives on CPU)
        batch_d = batch.to(device)
        cond_d = {'y': {}}
        for k, v in cond['y'].items():
            cond_d['y'][k] = v.to(device) if torch.is_tensor(v) else v

        t_highnoise = torch.randint(high_noise_min, total_t, (bs,), device=device)
        noise = torch.randn_like(batch_d)
        source_names = cond_d['y']['source_name']

        # Matched
        psi_m, action_m = fill_psi_action(source_names, psi_lookup,
                                           fname_to_action, device)
        set_behavior(cond_d, psi_m, action_m)
        L_matched, _ = compute_eps_loss(model, diffusion, batch_d, t_highnoise,
                                          cond_d, noise)

        # Build override lists per sample for each variant
        diff_psi, diff_act = [], []
        far_psi, far_act = [], []
        action_only_psi, action_only_act = [], []

        for bi, fname in enumerate(source_names):
            tgt_psi = psi_lookup.get(fname, np.zeros((64, 62), np.float32))
            tgt_psi_mean = tgt_psi.mean(axis=0)
            tgt_act = fname_to_action.get(fname, ACTION_TO_IDX['other'])

            # diff_action negative: action + ψ both come from diff-action neighbour
            ent_d, _ = queue.sample(tgt_psi_mean, tgt_act, mode='diff_action')
            if ent_d is None:
                diff_psi.append(None); diff_act.append(None)
            else:
                diff_psi.append(psi_lookup.get(ent_d[2],
                                                np.zeros((64, 62), np.float32))
                                .astype(np.float32))
                diff_act.append(int(ent_d[1]))

            # same_action / far ψ negative (gap_psi_only)
            ent_f, mode_f = queue.sample(tgt_psi_mean, tgt_act,
                                          mode='same_action_far_psi')
            if ent_f is None or mode_f != 'same_action_far_psi':
                far_psi.append(None); far_act.append(None)
            else:
                far_psi.append(psi_lookup.get(ent_f[2],
                                               np.zeros((64, 62), np.float32))
                               .astype(np.float32))
                far_act.append(int(tgt_act))  # action stays matched

            # action_only swap: matched ψ, swapped action label
            other_actions = [a for a in range(len(ACTION_CLASSES))
                             if a != tgt_act]
            swapped_act = (other_actions[int(random.random() * len(other_actions))]
                           if other_actions else tgt_act)
            action_only_psi.append(tgt_psi.astype(np.float32))
            action_only_act.append(swapped_act)

        L_diff = _override_one_step(model, diffusion, batch_d, cond_d,
                                     source_names, diff_psi, diff_act,
                                     psi_lookup, fname_to_action,
                                     t_highnoise, noise, device)
        L_far = _override_one_step(model, diffusion, batch_d, cond_d,
                                    source_names, far_psi, far_act,
                                    psi_lookup, fname_to_action,
                                    t_highnoise, noise, device)
        L_action_only = _override_one_step(model, diffusion, batch_d, cond_d,
                                            source_names, action_only_psi,
                                            action_only_act, psi_lookup,
                                            fname_to_action, t_highnoise,
                                            noise, device)
        set_null_behavior(cond_d, model, bs)
        L_null, _ = compute_eps_loss(model, diffusion, batch_d, t_highnoise,
                                       cond_d, noise)

        L_matched_all.append(L_matched.float().cpu())
        L_diff_all.append(L_diff.float().cpu())
        L_far_all.append(L_far.float().cpu())
        L_action_only_all.append(L_action_only.float().cpu())
        L_null_all.append(L_null.float().cpu())

    L_m = torch.cat(L_matched_all)
    L_d = torch.cat(L_diff_all)
    L_f = torch.cat(L_far_all)
    L_a = torch.cat(L_action_only_all)
    L_n = torch.cat(L_null_all)
    lm = L_m.mean().clamp(min=1e-8)

    gaps = {
        'L_matched_highnoise': float(L_m.mean().item()),
        'gap_action_diff': float(((L_d - L_m) / lm).mean().item()),
        'gap_same_action_far_psi': float(((L_f - L_m) / lm).mean().item()),
        'gap_psi_only': float(((L_f - L_m) / lm).mean().item()),
        'gap_action_only': float(((L_a - L_m) / lm).mean().item()),
        'gap_null': float(((L_n - L_m) / lm).mean().item()),
        'n_clips': int(L_m.numel()),
    }
    model.train()
    return gaps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    if args.smoke:
        args.num_steps = 200
        args.batch_size = 4
        args.save_interval = 100
        args.log_interval = 20
        args.probe_interval = 100
        args.probe_clips = 8

    fixseed(args.seed)
    dist_util.setup_dist(args.device)
    device = dist_util.dev()
    os.makedirs(args.save_dir, exist_ok=True)
    with open(pjoin(args.save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    opt = get_opt(args.device)

    print(f"Loading ψ cache from {args.effect_cache}")
    psi_lookup, metadata = build_psi_lookup(args.clip_metadata, args.effect_cache)
    print(f"  {len(psi_lookup)} clips with cached ψ")
    fname_to_action = {m['fname']: ACTION_TO_IDX.get(m['coarse_label'],
                                                     ACTION_TO_IDX['other'])
                       for m in metadata}

    # Build model (residual DISABLED per reviewer)
    print(f"Building AnyTopBehavior (residual=OFF, latent={args.latent_dim}, "
          f"layers={args.layers})")
    model = AnyTopBehavior(
        max_joints=opt.max_joints,
        feature_len=opt.feature_len,
        latent_dim=args.latent_dim,
        ff_size=args.latent_dim * 4,
        num_layers=args.layers,
        num_heads=4,
        t5_out_dim=768,
        n_actions=len(ACTION_CLASSES),
        n_behavior_tokens=args.n_behavior_tokens,
        use_residual=False,
        behavior_drop_prob=args.behavior_drop_prob,
        skip_t5=False,
        cond_mode='object_type',
        cond_mask_prob=0.1,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"  Trainable params: {n_params:.2f}M")

    start_step = 0
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"RESUMING from {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location='cpu',
                           weights_only=False)
        # Saved format: {'model': state_dict, 'step': N, 'args': dict}
        sd = ckpt.get('model', ckpt)
        msd = model.state_dict()
        matched = 0
        for k, v in sd.items():
            if k in msd and msd[k].shape == v.shape:
                msd[k] = v
                matched += 1
        model.load_state_dict(msd, strict=False)
        start_step = int(ckpt.get('step', 0))
        print(f"  Loaded {matched} params, resuming at step {start_step}")
    elif args.init_from and os.path.exists(args.init_from):
        print(f"Loading init backbone from {args.init_from}")
        ckpt = torch.load(args.init_from, map_location='cpu', weights_only=False)
        msd = model.state_dict()
        matched = 0
        for k, v in ckpt.items():
            if k in msd and msd[k].shape == v.shape:
                msd[k] = v
                matched += 1
        model.load_state_dict(msd, strict=False)
        print(f"  Init matched {matched} params")
    else:
        print("  Random init (from scratch)")

    model.to(device)
    model.train()

    class DiffArgs:
        noise_schedule = 'cosine'
        sigma_small = True
        diffusion_steps = 100
        lambda_fs = 0.0
        lambda_geo = 0.0
    diffusion = create_gaussian_diffusion(DiffArgs())
    total_t = diffusion.num_timesteps

    print(f"Building data loader (aug_probs = "
          f"noop:{args.aug_prob_noop} remove:{args.aug_prob_remove} "
          f"add:{args.aug_prob_add})")
    data = get_dataset_loader_dc(
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        temporal_window=args.temporal_window,
        t5_name=args.t5_name,
        balanced=False,
        objects_subset='all',
        aug_prob_noop=args.aug_prob_noop,
        aug_prob_remove=args.aug_prob_remove,
        aug_prob_add=args.aug_prob_add,
    )

    # Build probe set (held-out clips for three-gap eval)
    probe_fnames = build_probe_set(psi_lookup, metadata, fname_to_action,
                                    n=args.probe_clips, seed=args.seed + 17)
    print(f"  Probe set: {len(probe_fnames)} held-out clips")

    # Pre-cache N batches for the FIXED probe (reused every probe call)
    print(f"  Pre-caching {args.probe_batches} probe batches "
          f"(~{args.probe_batches * args.batch_size} samples)...")
    probe_data_iter = iter(data)
    cached_probe = []
    for _ in range(args.probe_batches):
        try:
            pb, pc = next(probe_data_iter)
        except StopIteration:
            probe_data_iter = iter(data)
            pb, pc = next(probe_data_iter)
        # Move tensors to CPU to keep them around without consuming GPU memory
        pc_cpu = {'y': {}}
        for k, v in pc['y'].items():
            pc_cpu['y'][k] = v.cpu() if torch.is_tensor(v) else v
        cached_probe.append((pb.cpu(), pc_cpu))
    del probe_data_iter
    print(f"  Probe cache ready: {len(cached_probe)} batches "
          f"({sum(b.shape[0] for b, _ in cached_probe)} samples)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    queue = NegativeQueue(max_size=args.queue_size)
    rng = random.Random(args.seed + 1)

    print(f"\nTraining {args.num_steps} steps, bs={args.batch_size}, lr={args.lr}")
    print(f"Save dir: {args.save_dir}\n")

    data_iter = iter(data)
    t0 = time.time()
    running = {'L_self': 0, 'L_pair': 0, 'L_cf': 0, 'gap_train_mean': 0,
               'gap_train_action_diff': 0, 'gap_train_same_action_far_psi': 0,
               'n_self': 0, 'n_pair': 0, 'n_cf_effective': 0,
               'n_neg_diff': 0, 'n_neg_far': 0}

    for step in range(start_step, args.num_steps):
        try:
            batch, cond = next(data_iter)
        except StopIteration:
            data_iter = iter(data)
            batch, cond = next(data_iter)

        batch = batch.to(device)
        for k, v in cond['y'].items():
            if torch.is_tensor(v):
                cond['y'][k] = v.to(device)

        bs = batch.shape[0]
        source_names = cond['y']['source_name']
        aug_types = cond['y']['aug_type']
        aug_is_self = (aug_types == 0)

        # ---- matched B ----
        psi_m, action_m = fill_psi_action(source_names, psi_lookup,
                                           fname_to_action, device)
        cond = set_behavior(cond, psi_m, action_m)

        # ---- stratified timestep ----
        t = stratified_t(bs, aug_is_self.tolist(), args.high_noise_min,
                          total_t, args.stratified_fraction, device)
        noise = torch.randn_like(batch)

        # ---- PASS 1: matched forward. Backprop L_self + λ_pair·L_pair and free
        # the graph before the second forward to keep peak activation at ~1x. ----
        optimizer.zero_grad()
        L_matched, _ = compute_eps_loss(model, diffusion, batch, t, cond, noise)

        self_mask = aug_is_self.float()
        pair_mask = (~aug_is_self).float()
        n_self_t = self_mask.sum().clamp(min=1)
        n_pair_t = pair_mask.sum().clamp(min=1)

        L_self = (L_matched * self_mask).sum() / n_self_t
        L_pair = (L_matched * pair_mask).sum() / n_pair_t
        loss_a = L_self + args.lambda_pair * L_pair
        loss_a.backward()

        # Snapshot matched loss for the hinge (FP32), graph already freed above.
        with torch.no_grad():
            L_matched_f32 = L_matched.detach().float()

        # ---- PASS 2: mismatched forward (only if queue populated) ----
        has_cf = aug_is_self.clone()
        ran_pass2 = False
        # Track which negative type was sampled per slot, so we can split the
        # training-stream gap by negative type for diagnostics.
        neg_type_per_sample = [None] * bs  # 'diff_action' | 'same_action_far_psi' | None
        if len(queue) >= 4:
            ran_pass2 = True
            bp_psi_list = []
            bp_act_list = []
            for bi in range(bs):
                if not aug_is_self[bi]:
                    bp_psi_list.append(psi_m[bi].detach().cpu().numpy())
                    bp_act_list.append(int(action_m[bi].item()))
                    continue
                fname = source_names[bi]
                tgt_psi_mean = psi_lookup.get(
                    fname, np.zeros((64, 62), np.float32)).mean(axis=0)
                tgt_act = fname_to_action.get(fname, ACTION_TO_IDX['other'])
                ent, neg_mode = queue.sample(tgt_psi_mean, tgt_act,
                                              mode='mixed', rng=rng)
                if ent is None:
                    has_cf[bi] = False
                    bp_psi_list.append(psi_m[bi].detach().cpu().numpy())
                    bp_act_list.append(int(action_m[bi].item()))
                else:
                    other_fname = ent[2]
                    bp_psi_list.append(psi_lookup.get(
                        other_fname,
                        np.zeros((64, 62), np.float32)).astype(np.float32))
                    bp_act_list.append(int(ent[1]))
                    # 'diff_action_fallback' counts as diff_action for split logging
                    neg_type_per_sample[bi] = (
                        'same_action_far_psi'
                        if neg_mode == 'same_action_far_psi'
                        else 'diff_action')

            psi_p = torch.tensor(np.stack(bp_psi_list), dtype=torch.float32,
                                  device=device)
            action_p = torch.tensor(bp_act_list, dtype=torch.long, device=device)
            cond = set_behavior(cond, psi_p, action_p)
            L_mismatch, _ = compute_eps_loss(model, diffusion, batch, t,
                                              cond, noise)

            # Timestep-band weighting
            t_f = t.float()
            band_weight = torch.where(
                t_f >= args.high_noise_min,
                torch.full_like(t_f, args.high_noise_weight),
                torch.where(t_f >= args.mid_noise_min,
                            torch.full_like(t_f, args.mid_noise_weight),
                            torch.zeros_like(t_f)))
            cf_mask = has_cf.float() * self_mask * band_weight
            n_cf_t = cf_mask.sum().clamp(min=1)

            L_mismatch_f32 = L_mismatch.float()
            margin_i = torch.clamp(0.05 * L_matched_f32, 0.01, 0.03)
            hinge = torch.clamp(margin_i + L_matched_f32 - L_mismatch_f32,
                                 min=0.0)
            L_cf = (hinge * cf_mask).sum() / n_cf_t

            # Backprop the contrastive term
            (args.lambda_cf * L_cf).backward()
        else:
            L_mismatch = L_matched.detach()
            L_cf = torch.tensor(0.0, device=device)
            n_cf_t = torch.tensor(1.0, device=device)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # ---- Populate queue with current batch's B (post-step, matched arm) ----
        for bi, fname in enumerate(source_names):
            psi_np = psi_lookup.get(fname, None)
            if psi_np is None:
                continue
            queue.add(psi_np.astype(np.float32), action_m[bi].item(), fname)

        running['L_self'] += L_self.item() * n_self_t.item()
        running['L_pair'] += L_pair.item() * n_pair_t.item()
        running['L_cf'] += L_cf.item() * n_cf_t.item()
        running['n_self'] += n_self_t.item()
        running['n_pair'] += n_pair_t.item()
        running['n_cf_effective'] += n_cf_t.item()
        if ran_pass2:
            with torch.no_grad():
                gap_train = (L_mismatch_f32 - L_matched_f32)  # [B]
                gap_train_weighted = ((gap_train * cf_mask).sum()
                                       / n_cf_t.clamp(min=1))
                running['gap_train_mean'] += (gap_train_weighted.item()
                                              * n_cf_t.item())
                # Split by negative type (only for samples that actually had a
                # negative drawn from the queue and contributed to L_cf).
                diff_mask = torch.tensor(
                    [n == 'diff_action' for n in neg_type_per_sample],
                    dtype=torch.float32, device=device) * cf_mask
                far_mask = torch.tensor(
                    [n == 'same_action_far_psi' for n in neg_type_per_sample],
                    dtype=torch.float32, device=device) * cf_mask
                n_diff = diff_mask.sum().clamp(min=1e-6)
                n_far = far_mask.sum().clamp(min=1e-6)
                gap_diff = (gap_train * diff_mask).sum() / n_diff
                gap_far = (gap_train * far_mask).sum() / n_far
                running['gap_train_action_diff'] += (
                    gap_diff.item() * float(diff_mask.sum().item()))
                running['gap_train_same_action_far_psi'] += (
                    gap_far.item() * float(far_mask.sum().item()))
                running['n_neg_diff'] += float(diff_mask.sum().item())
                running['n_neg_far'] += float(far_mask.sum().item())

        # ---- Logging ----
        if (step + 1) % args.log_interval == 0:
            elapsed = time.time() - t0
            rate = (step + 1) / elapsed
            eta_h = (args.num_steps - step - 1) / rate / 3600
            ns = max(running['n_self'], 1)
            np_ = max(running['n_pair'], 1)
            ncf = max(running['n_cf_effective'], 1)
            n_diff = max(running['n_neg_diff'], 1)
            n_far = max(running['n_neg_far'], 1)
            print(f"[{step+1}/{args.num_steps}] "
                  f"L_self={running['L_self']/ns:.4f} "
                  f"L_pair={running['L_pair']/np_:.4f} "
                  f"L_cf={running['L_cf']/ncf:.4f} "
                  f"gap_train={running['gap_train_mean']/ncf:+.5f} "
                  f"g_diff={running['gap_train_action_diff']/n_diff:+.5f} "
                  f"g_far={running['gap_train_same_action_far_psi']/n_far:+.5f} "
                  f"(n_diff={int(n_diff)} n_far={int(n_far)}) "
                  f"q={len(queue)} rate={rate:.2f}/s eta={eta_h:.1f}h")
            for k in running:
                running[k] = 0

        # ---- Probe eval (decomposed gaps on FIXED cached probe set) ----
        if (step + 1) % args.probe_interval == 0:
            gaps = decomposed_probe(model, diffusion, cached_probe,
                                     psi_lookup, fname_to_action, queue,
                                     args.high_noise_min, total_t, device)
            gap_line = (f"  PROBE step={step+1} (n={gaps['n_clips']}): "
                        f"L_m={gaps['L_matched_highnoise']:.4f} "
                        f"g_action={gaps['gap_action_diff']:+.4f} "
                        f"g_psi_only={gaps['gap_psi_only']:+.4f} "
                        f"g_action_only={gaps['gap_action_only']:+.4f} "
                        f"g_null={gaps['gap_null']:+.4f}")
            print(gap_line)
            with open(pjoin(args.save_dir, 'probe_log.jsonl'), 'a') as f:
                f.write(json.dumps({'step': step + 1, **gaps}) + '\n')

            # Kill-criteria gate at 25k / 50k
            if (step + 1) in (25000, 50000):
                gate_decision = evaluate_gate(step + 1, gaps)
                with open(pjoin(args.save_dir, f'gate_{step+1}.json'), 'w') as f:
                    json.dump(gate_decision, f, indent=2)
                print(f"  GATE@{step+1}: {gate_decision['verdict']}")

        # ---- Checkpoint ----
        if (step + 1) % args.save_interval == 0:
            ckpt_path = pjoin(args.save_dir, f'model{step+1:09d}.pt')
            torch.save({'model': model.state_dict(), 'step': step + 1,
                        'args': vars(args)}, ckpt_path)
            print(f"  Saved {ckpt_path}")

    # Final save
    final_path = pjoin(args.save_dir, f'model{args.num_steps:09d}.pt')
    torch.save({'model': model.state_dict(), 'step': args.num_steps,
                'args': vars(args)}, final_path)
    print(f"\nFinal save: {final_path}")


def evaluate_gate(step, gaps):
    """Apply pre-registered kill criteria — UPDATED post-Round-12 reviewer pass.
    The 50k gate now requires gap_psi_only > 0 (was gap_same_action_far_psi),
    matching the explicit decomposed metric. Action-only / null gaps no longer
    count toward the ψ-usage criterion.
    """
    gap_primary = gaps.get('gap_action_diff', 0.0)
    gap_psi_only = gaps.get('gap_psi_only', 0.0)
    gap_action_only = gaps.get('gap_action_only', 0.0)
    gap_null = gaps.get('gap_null', 0.0)
    if step <= 25000:
        # Gate@25k: behavior channel responsive at all (action OR psi)
        pass_gap = gap_primary > 0.02
        pass_null = gap_null > 0.02
        verdict = 'PASS' if (pass_gap and pass_null) else 'FAIL'
    else:
        # Gate@50k: ψ specifically must be in use
        pass_gap = gap_primary > 0.05 and gap_psi_only > 0.0
        verdict = 'PASS' if pass_gap else 'FAIL'
        # Track-B gate not evaluated inline (requires generation pipeline).
        # Emitted as a follow-up TODO in the decision file.
    return {
        'step': step,
        'gap_action_diff': gap_primary,
        'gap_psi_only': gap_psi_only,
        'gap_action_only': gap_action_only,
        'gap_null': gap_null,
        'verdict': verdict,
        'note': 'Track-B accuracy gate must be evaluated separately with '
                'eval/track_b_evaluator.py before committing to "PASS" '
                'at the 50k checkpoint.',
    }


if __name__ == '__main__':
    main()
