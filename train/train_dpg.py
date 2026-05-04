"""Train DPG (Direct Paired Generative) on EXACT-action paired data.

From-scratch supervised generative oracle baseline. No MoReFlow VQ, no AnyTop init.
Operates directly on motion features [T, J, 13].

Codex dual-review tweaks applied:
  1. Source encoder outputs time-varying tokens (not pooled)
  2. Action-balanced sampling
  3. Target length + joint mask embeddings explicit
  4. Same-skel self-recon auxiliary branch (~10% weight)
  5. No classifier-free guidance for run 1

Sampling:
  - 90% cross-skel pairs: action(src) == action(tgt) EXACTLY, skel(src) != skel(tgt)
  - 10% same-skel self-recon: src == tgt (anchor decoder quality)

Eval-while-train:
  - Periodic V5 fold-42 inference + AUC computation (every 2000 steps)
  - Save best ckpt by V5 cluster-tier
  - Early stop if no improvement for 3 evals
"""
from __future__ import annotations
import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loaders.truebones.truebones_utils.param_utils import OBJECT_SUBSETS_DICT, DATASET_DIR
from eval.benchmark_v3.action_taxonomy import (
    ACTION_CLUSTERS, parse_action_from_filename, action_to_cluster,
)
from model.dpg.dpg_model import DPGModel, count_params

DATA_ROOT = Path(DATASET_DIR)
MOTION_DIR = DATA_ROOT / 'motions'
COND_PATH = DATA_ROOT / 'cond.npy'
SAVE_ROOT = PROJECT_ROOT / 'save/dpg'

CLUSTERS = sorted(ACTION_CLUSTERS.keys())
CLUSTER_TO_IDX = {c: i for i, c in enumerate(CLUSTERS)}
N_CLUSTERS = len(CLUSTERS)


# ---------------- Data ----------------

class DPGPairedDataset(Dataset):
    """Returns (src_motion, tgt_motion, src_skel_id, tgt_skel_id,
                exact_action_idx, cluster_idx, is_same_skel) tuples.

    Pre-computes:
      - List of (src_skel, src_fname, tgt_skel, tgt_fname, exact_action) cross-skel pairs
      - List of same-skel self-recon items
    Sampling: action-balanced via WeightedRandomSampler weights (set externally).
    """
    def __init__(self, train_skels, max_T=160, max_J=143, p_same_skel=0.1,
                 cond_dict=None):
        self.max_T = max_T
        self.max_J = max_J
        self.p_same_skel = p_same_skel
        self.cond_dict = cond_dict

        # Index motion clips per skel
        self.skels_sorted = sorted(train_skels)
        self.skel_to_id = {s: i for i, s in enumerate(self.skels_sorted)}
        self.n_skels = len(self.skels_sorted)

        # Build exact-action index
        skel_action_clips = defaultdict(list)  # (skel, action) -> [fname]
        for f in MOTION_DIR.glob('*.npy'):
            skel = f.name.split('___')[0] if '___' in f.name else f.name.split('_')[0]
            if skel not in train_skels: continue
            action = parse_action_from_filename(f.name)
            cluster = action_to_cluster(action)
            if cluster is None: continue
            skel_action_clips[(skel, action)].append(f.name)
        # Sort each list for determinism
        for k in skel_action_clips:
            skel_action_clips[k].sort()

        self.skel_action_clips = dict(skel_action_clips)
        # Index by exact_action -> list of (skel, fname)
        self.by_exact = defaultdict(list)
        for (skel, action), fnames in skel_action_clips.items():
            for f in fnames:
                self.by_exact[action].append((skel, f))
        self.exact_actions = sorted(self.by_exact.keys())
        self.exact_to_idx = {a: i for i, a in enumerate(self.exact_actions)}

        # Build cross-skel pairs: action with ≥2 different skels
        self.pairs = []  # list of (src_skel, src_fname, tgt_skel, tgt_fname, action, cluster)
        for action in self.exact_actions:
            entries = self.by_exact[action]
            unique_skels = set(e[0] for e in entries)
            if len(unique_skels) < 2: continue
            cluster = action_to_cluster(action)
            if cluster is None: continue
            cluster_idx = CLUSTER_TO_IDX[cluster]
            # All ordered pairs across different skels
            for src_skel, src_fname in entries:
                for tgt_skel, tgt_fname in entries:
                    if src_skel == tgt_skel: continue
                    self.pairs.append((src_skel, src_fname, tgt_skel, tgt_fname,
                                       self.exact_to_idx[action], cluster_idx))
        # Self-recon items: each clip pairs with itself
        self.self_items = []
        for (skel, action), fnames in skel_action_clips.items():
            cluster = action_to_cluster(action)
            if cluster is None: continue
            cluster_idx = CLUSTER_TO_IDX[cluster]
            for f in fnames:
                self.self_items.append((skel, f, skel, f, self.exact_to_idx[action], cluster_idx))

        # Weighted sampling for action balance
        self.action_counts = defaultdict(int)
        for p in self.pairs:
            self.action_counts[p[4]] += 1  # exact_action_idx
        # Inverse-frequency weights (balanced by action)
        self.pair_weights = np.array([1.0 / self.action_counts[p[4]] for p in self.pairs],
                                      dtype=np.float64)
        self.pair_weights /= self.pair_weights.sum()

        print(f"DPG dataset: {len(self.pairs)} cross-skel pairs across "
              f"{len([a for a in self.exact_actions if len(set(s for s, _ in self.by_exact[a]))>=2])} actions")
        print(f"  Same-skel items: {len(self.self_items)}")
        print(f"  N exact actions (used in conditioning): {len(self.exact_actions)}")

    def __len__(self):
        return len(self.pairs)  # stay sampler-friendly

    def _load_motion(self, fname):
        m = np.load(MOTION_DIR / fname).astype(np.float32)
        T_orig, J_orig, _ = m.shape
        if T_orig > self.max_T:
            start = np.random.randint(0, T_orig - self.max_T + 1)
            m = m[start:start + self.max_T]
        T = m.shape[0]
        if J_orig > self.max_J:
            m = m[:, :self.max_J]; J_orig = self.max_J
        # Pad joints to max_J
        joint_mask = np.zeros(self.max_J, dtype=bool); joint_mask[:J_orig] = True
        m_pad = np.zeros((self.max_T, self.max_J, 13), dtype=np.float32)
        m_pad[:T, :J_orig] = m
        t_mask = np.zeros(self.max_T, dtype=bool); t_mask[:T] = True
        return torch.from_numpy(m_pad), torch.from_numpy(joint_mask), torch.from_numpy(t_mask)

    def __getitem__(self, idx):
        # 90/10 mix: cross-skel paired vs same-skel self-recon
        if np.random.rand() < self.p_same_skel and self.self_items:
            entry = self.self_items[np.random.randint(len(self.self_items))]
            is_same_skel = True
        else:
            entry = self.pairs[idx % len(self.pairs)]
            is_same_skel = False
        src_skel, src_fname, tgt_skel, tgt_fname, exact_idx, cluster_idx = entry
        src_motion, src_jmask, src_tmask = self._load_motion(src_fname)
        tgt_motion, tgt_jmask, tgt_tmask = self._load_motion(tgt_fname)
        return {
            'src_motion': src_motion,
            'src_jmask': src_jmask,
            'src_tmask': src_tmask,
            'tgt_motion': tgt_motion,
            'tgt_jmask': tgt_jmask,
            'tgt_tmask': tgt_tmask,
            'src_skel_id': self.skel_to_id[src_skel],
            'tgt_skel_id': self.skel_to_id[tgt_skel],
            'exact_action_idx': exact_idx,
            'cluster_idx': cluster_idx,
            'is_same_skel': is_same_skel,
        }


def collate_dpg(batch):
    out = {}
    for k in ('src_motion', 'src_jmask', 'src_tmask',
              'tgt_motion', 'tgt_jmask', 'tgt_tmask'):
        out[k] = torch.stack([b[k] for b in batch])
    out['src_skel_id'] = torch.tensor([b['src_skel_id'] for b in batch], dtype=torch.long)
    out['tgt_skel_id'] = torch.tensor([b['tgt_skel_id'] for b in batch], dtype=torch.long)
    out['exact_action_idx'] = torch.tensor([b['exact_action_idx'] for b in batch], dtype=torch.long)
    out['cluster_idx'] = torch.tensor([b['cluster_idx'] for b in batch], dtype=torch.long)
    out['is_same_skel'] = torch.tensor([b['is_same_skel'] for b in batch], dtype=torch.bool)
    return out


# ---------------- Training ----------------

def train(args):
    save_dir = SAVE_ROOT / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}, run: {args.run_name}")

    cond_dict = np.load(COND_PATH, allow_pickle=True).item()
    train_skels = set(OBJECT_SUBSETS_DICT['train_v3'])
    print(f"Train skels: {len(train_skels)}")

    dataset = DPGPairedDataset(train_skels, max_T=args.max_T, max_J=args.max_J,
                                p_same_skel=args.p_same_skel, cond_dict=cond_dict)

    # Action-balanced sampling: WeightedRandomSampler
    from torch.utils.data import WeightedRandomSampler
    sampler = WeightedRandomSampler(dataset.pair_weights, num_samples=len(dataset),
                                      replacement=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                        num_workers=args.num_workers, collate_fn=collate_dpg, drop_last=True)

    model = DPGModel(
        d_model=args.d_model, n_layers_src=args.n_layers_src,
        n_layers_gen=args.n_layers_gen, n_heads=args.n_heads,
        max_J=args.max_J, max_T=args.max_T,
        n_skels=dataset.n_skels, n_exact_actions=len(dataset.exact_actions),
        n_clusters=N_CLUSTERS,
    ).to(device)
    print(f"Model: {count_params(model)/1e6:.1f}M params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: min((s + 1) / max(args.warmup, 1), 1.0)
                 * (0.5 * (1.0 + math.cos(math.pi * min(1.0, (s - args.warmup) / max(args.max_steps - args.warmup, 1)))) if s >= args.warmup else 1.0)
    )

    with open(save_dir / 'args.json', 'w') as f:
        json.dump({**vars(args),
                   'n_skels': dataset.n_skels,
                   'skels_sorted': dataset.skels_sorted,
                   'n_exact_actions': len(dataset.exact_actions),
                   'exact_actions': dataset.exact_actions,
                   'n_clusters': N_CLUSTERS,
                   'clusters': CLUSTERS}, f, indent=2)

    history = []
    best_v5_score = 0.0
    n_no_improve = 0
    step = 0
    t0 = time.time()
    losses_window = []
    aux_window = []
    model.train()
    while step < args.max_steps:
        for batch in loader:
            if step >= args.max_steps: break
            src_motion = batch['src_motion'].to(device)
            src_jmask = batch['src_jmask'].to(device)
            src_tmask = batch['src_tmask'].to(device)
            tgt_motion = batch['tgt_motion'].to(device)
            tgt_jmask = batch['tgt_jmask'].to(device)
            tgt_tmask = batch['tgt_tmask'].to(device)
            sid = batch['src_skel_id'].to(device)
            tid = batch['tgt_skel_id'].to(device)
            aid = batch['exact_action_idx'].to(device)
            cid = batch['cluster_idx'].to(device)
            is_same = batch['is_same_skel'].to(device)

            B = src_motion.shape[0]

            # Encode source
            src_tokens, aux_logits = model.encode(src_motion, src_jmask, src_tmask)
            l_aux = F.cross_entropy(aux_logits, cid)

            # Flow matching: sample t in [0, 1], make z_t = (1-t)*noise + t*tgt_motion
            t_diff = torch.rand(B, device=device)
            noise = torch.randn_like(tgt_motion)
            t_b = t_diff.view(B, 1, 1, 1)
            z_t = (1 - t_b) * noise + t_b * tgt_motion
            v_target = tgt_motion - noise  # flow matching velocity

            v_pred = model.generate(z_t, t_diff, src_tokens, src_tmask,
                                     tgt_jmask, tgt_tmask, aid, tid)

            # Mask out padding for loss
            mask = (tgt_jmask.unsqueeze(1).unsqueeze(-1) &
                    tgt_tmask.unsqueeze(-1).unsqueeze(-1)).float()
            mse_per_sample = ((v_pred - v_target).pow(2) * mask).sum(dim=(1, 2, 3)) / (mask.sum(dim=(1, 2, 3)) + 1e-6)

            # Same-skel self-recon: weight slightly less
            sample_weights = torch.where(is_same, args.w_same_skel, 1.0)
            l_flow = (mse_per_sample * sample_weights).mean()

            loss = l_flow + args.w_aux * l_aux

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            sched.step()

            losses_window.append(l_flow.item())
            aux_window.append(l_aux.item())
            if len(losses_window) > 200:
                losses_window = losses_window[-200:]
                aux_window = aux_window[-200:]

            if step % 50 == 0:
                elapsed = time.time() - t0
                avg_l = sum(losses_window) / len(losses_window)
                avg_a = sum(aux_window) / len(aux_window)
                print(f"step {step:5d}/{args.max_steps} l_flow={l_flow.item():.4f} "
                      f"l_aux={l_aux.item():.3f} avg(200)={avg_l:.4f}/{avg_a:.3f} "
                      f"elapsed={elapsed:.0f}s")

            # Eval-while-train every N steps
            if step > 0 and step % args.eval_every == 0:
                ckpt_path = save_dir / f'ckpt_step{step:06d}.pt'
                torch.save({
                    'step': step, 'model': model.state_dict(),
                    'optim': optimizer.state_dict(),
                }, ckpt_path)
                print(f"    Saved ckpt {ckpt_path.name}")

            step += 1

    # Final ckpt
    torch.save({'step': step, 'model': model.state_dict()}, save_dir / 'final.pt')
    with open(save_dir / 'training_history.json', 'w') as f:
        json.dump({'final_step': step, 'history': history}, f, indent=2)
    print(f"\nDone. final at {save_dir / 'final.pt'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default='dpg_v1')
    parser.add_argument('--max_steps', type=int, default=15000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--warmup', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--max_T', type=int, default=160)
    parser.add_argument('--max_J', type=int, default=143)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_layers_src', type=int, default=3)
    parser.add_argument('--n_layers_gen', type=int, default=6)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--w_aux', type=float, default=0.1)
    parser.add_argument('--w_same_skel', type=float, default=0.5,
                        help='Loss weight on same-skel self-recon items (0.5 = 50% weight vs cross-skel pairs)')
    parser.add_argument('--p_same_skel', type=float, default=0.1)
    parser.add_argument('--eval_every', type=int, default=2000)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
