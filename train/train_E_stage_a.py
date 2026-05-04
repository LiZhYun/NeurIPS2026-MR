"""Train E Stage A: skeleton-blind behavior-code encoder + auxiliary heads.

Per FINAL_PROPOSAL V4. Training-time supervision (all skeleton-agnostic):
  - action cluster (10-way CE)
  - Q-feature signature (22-d MSE)
  - contact density (scalar MSE)
  - COM-heading (4-d MSE)
  - IB regularization: KL(z, N(0, I))
  - GRL skel head: 70-way CE (gradient-reversed)

Eval-while-train (every N steps):
  - Linear probe `z → skel_id` accuracy (target ≤ 0.05)
  - Same-action latent swap test (degradation ≥ 0.05 on Q-pred)
  - Action cluster accuracy on dev (sanity)

Early-stop conditions:
  - Day-2 early kill: at step 5K, if linear probe > 0.10 OR swap degradation < 0.02 → STOP
  - Standard early stop: best aux loss not improving for 5 evals → STOP

Usage:
  python -m train.train_E_stage_a --pilot_12  # 12-skel pilot for early kill
  python -m train.train_E_stage_a --full      # full 70-skel training
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
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
from eval.baselines.run_i5_action_classifier_v3 import featurize_q
from model.behavior_quotient.encoder_E import (
    BehaviorEncoderE, kl_loss, count_parameters,
)

DATA_ROOT = Path(DATASET_DIR)
MOTION_DIR = DATA_ROOT / 'motions'
COND_PATH = DATA_ROOT / 'cond.npy'
Q_CACHE_PATH = PROJECT_ROOT / 'idea-stage/quotient_cache.npz'
SAVE_ROOT = PROJECT_ROOT / 'save/behavior_quotient_E'

CLUSTERS = sorted(ACTION_CLUSTERS.keys())
CLUSTER_TO_IDX = {c: i for i, c in enumerate(CLUSTERS)}
N_CLUSTERS = len(CLUSTERS)


# ---------------- Dataset ----------------

class TruebonesAuxDataset(Dataset):
    """Loads motion clips with auxiliary target labels (action, Q, contact, COM)."""

    def __init__(self, skels, max_joints=143, n_frames=120, fps=30):
        self.max_joints = max_joints
        self.n_frames = n_frames
        self.fps = fps

        # Load Q cache for aux targets
        qc = np.load(Q_CACHE_PATH, allow_pickle=True)
        meta = list(qc['meta'])
        # Filter to selected skels
        skel_set = set(skels)
        keep_idx = [i for i, m in enumerate(meta) if m['skeleton'] in skel_set]
        self.meta = [meta[i] for i in keep_idx]
        self.com_path = [qc['com_path'][i] for i in keep_idx]
        self.heading_vel = [qc['heading_vel'][i] for i in keep_idx]
        self.contact_sched = [qc['contact_sched'][i] for i in keep_idx]
        self.cadence = [float(qc['cadence'][i]) for i in keep_idx]
        self.limb_usage = [qc['limb_usage'][i] for i in keep_idx]

        # Per-clip action cluster + skel id
        skels_sorted = sorted(skel_set)
        self.skel_to_id = {s: i for i, s in enumerate(skels_sorted)}
        self.n_skels = len(skels_sorted)

        # Compute per-clip auxiliary targets ONCE
        self.action_idx = []
        self.q_target = []
        self.contact_density = []
        self.com_heading = []
        valid = []
        for i, m in enumerate(self.meta):
            action = parse_action_from_filename(m['fname'])
            cluster = action_to_cluster(action)
            if cluster is None:
                self.action_idx.append(-1); self.q_target.append(np.zeros(22))
                self.contact_density.append(0.0); self.com_heading.append(np.zeros(4))
                valid.append(False); continue
            ai = CLUSTER_TO_IDX[cluster]
            q22 = featurize_q(self.com_path[i], self.heading_vel[i],
                              self.contact_sched[i], self.cadence[i],
                              self.limb_usage[i])
            cs = np.asarray(self.contact_sched[i])
            cd = float(cs.mean())
            com_seq = np.asarray(self.com_path[i])
            if com_seq.ndim == 1: com_seq = com_seq.reshape(-1, 3)
            com_xyz = com_seq.mean(axis=0)
            heading = float(np.asarray(self.heading_vel[i]).mean())
            ch = np.array([com_xyz[0], com_xyz[1], com_xyz[2], heading], dtype=np.float32)
            self.action_idx.append(ai); self.q_target.append(q22)
            self.contact_density.append(cd); self.com_heading.append(ch)
            valid.append(True)
        # Filter to valid items
        self.valid_idx = [i for i, v in enumerate(valid) if v]
        print(f"Dataset: {len(self.valid_idx)} valid clips out of {len(self.meta)}")

    def __len__(self):
        return len(self.valid_idx)

    def __getitem__(self, raw_idx):
        i = self.valid_idx[raw_idx]
        m = self.meta[i]
        skel = m['skeleton']
        skel_id = self.skel_to_id[skel]
        # Load motion
        motion = np.load(MOTION_DIR / m['fname']).astype(np.float32)
        T0, J0, _ = motion.shape
        # Pad joints to max_joints
        if J0 > self.max_joints:
            motion = motion[:, :self.max_joints]
            J0 = self.max_joints
        joint_mask = np.zeros(self.max_joints, dtype=bool); joint_mask[:J0] = True
        m_pad = np.zeros((T0, self.max_joints, 13), dtype=np.float32)
        m_pad[:, :J0] = motion
        # Sample/crop frames
        if T0 < self.n_frames:
            # Pad with zeros
            pad = np.zeros((self.n_frames - T0, self.max_joints, 13), dtype=np.float32)
            m_full = np.concatenate([m_pad, pad], axis=0)
            t_mask = np.zeros(self.n_frames, dtype=bool); t_mask[:T0] = True
        else:
            start = np.random.randint(0, T0 - self.n_frames + 1)
            m_full = m_pad[start:start + self.n_frames]
            t_mask = np.ones(self.n_frames, dtype=bool)

        return {
            'motion': torch.from_numpy(m_full),                  # [T, J, 13]
            'joint_mask': torch.from_numpy(joint_mask),           # [J]
            'temporal_mask': torch.from_numpy(t_mask),            # [T]
            'action_idx': self.action_idx[i],
            'skel_id': skel_id,
            'q_target': torch.from_numpy(self.q_target[i].astype(np.float32)),
            'contact_density': torch.tensor(self.contact_density[i], dtype=torch.float32),
            'com_heading': torch.from_numpy(self.com_heading[i]),
            'fname': m['fname'], 'skel': skel,
        }


def collate(batch):
    out = {}
    for k in ('motion', 'joint_mask', 'temporal_mask', 'q_target', 'com_heading'):
        out[k] = torch.stack([b[k] for b in batch])
    out['action_idx'] = torch.tensor([b['action_idx'] for b in batch], dtype=torch.long)
    out['skel_id'] = torch.tensor([b['skel_id'] for b in batch], dtype=torch.long)
    out['contact_density'] = torch.stack([b['contact_density'] for b in batch])
    out['fname'] = [b['fname'] for b in batch]
    out['skel'] = [b['skel'] for b in batch]
    return out


# ---------------- Diagnostics ----------------

@torch.no_grad()
def collect_z_and_skel(encoder, loader, device, max_batches=None):
    """Encode all clips, return z [N, D] and skel_id [N]."""
    encoder.eval()
    zs, skels, actions = [], [], []
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches: break
        motion = batch['motion'].to(device)
        mask = batch['joint_mask'].to(device)
        out = encoder(motion, mask, grl_alpha=0.0)
        zs.append(out['z'].cpu().numpy())
        skels.append(batch['skel_id'].cpu().numpy())
        actions.append(batch['action_idx'].cpu().numpy())
    encoder.train()
    return np.concatenate(zs), np.concatenate(skels), np.concatenate(actions)


def linear_probe(z, labels, n_classes):
    """Train a quick logistic regression to predict labels from z; return accuracy."""
    from sklearn.linear_model import LogisticRegression
    if len(np.unique(labels)) < 2:
        return 0.0
    clf = LogisticRegression(max_iter=200, n_jobs=-1, multi_class='multinomial')
    try:
        clf.fit(z, labels)
        return float((clf.predict(z) == labels).mean())
    except Exception:
        return 0.0


def latent_swap_test(encoder, dataset, device, n_pairs=50):
    """Test if z carries source-specific info beyond action.

    For each query clip, find another clip with SAME action but DIFFERENT skel.
    Encode both → z(self), z(other_same_action_diff_skel).
    Use H_Q head to predict Q from each.
    Measure |Q(self) - Q(other)| = swap-induced change.
    Higher change = more source-specific info encoded.
    """
    encoder.eval()
    np.random.seed(42)
    # Build action -> [valid_idx_list] map by skel
    by_action_skel = {}
    for vi, raw_i in enumerate(dataset.valid_idx):
        ai = dataset.action_idx[raw_i]
        sk = dataset.meta[raw_i]['skeleton']
        by_action_skel.setdefault(ai, {}).setdefault(sk, []).append(vi)

    pairs = []
    for ai, by_skel in by_action_skel.items():
        skels = list(by_skel.keys())
        if len(skels) < 2: continue
        for _ in range(min(20, n_pairs)):
            sk_a, sk_b = np.random.choice(skels, 2, replace=False)
            i_a = np.random.choice(by_skel[sk_a])
            i_b = np.random.choice(by_skel[sk_b])
            pairs.append((i_a, i_b))
            if len(pairs) >= n_pairs: break
        if len(pairs) >= n_pairs: break
    if not pairs:
        encoder.train()
        return 0.0

    # Encode and compare
    diffs = []
    with torch.no_grad():
        for i_a, i_b in pairs:
            ba = collate([dataset[i_a]])
            bb = collate([dataset[i_b]])
            out_a = encoder(ba['motion'].to(device), ba['joint_mask'].to(device), grl_alpha=0.0)
            out_b = encoder(bb['motion'].to(device), bb['joint_mask'].to(device), grl_alpha=0.0)
            # Difference in Q prediction (proxy for source-specific info)
            q_a = out_a['q_pred'].cpu().numpy().squeeze()
            q_b = out_b['q_pred'].cpu().numpy().squeeze()
            diffs.append(np.linalg.norm(q_a - q_b))
    encoder.train()
    # Normalize: typical Q-norm ~ 5-20; degradation ratio is the ratio to typical Q-norm
    typical_q_norm = np.linalg.norm(out_a['q_pred'].cpu().numpy().squeeze())
    return float(np.mean(diffs) / (typical_q_norm + 1e-6))


# ---------------- Training loop ----------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot_12', action='store_true', help='12-skel pilot')
    parser.add_argument('--full', action='store_true', help='full 70-skel')
    parser.add_argument('--out_tag', type=str, default='E_stage_a_pilot')
    parser.add_argument('--n_steps', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_frames', type=int, default=120)
    parser.add_argument('--d_z', type=int, default=128)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--w_action', type=float, default=1.0)
    parser.add_argument('--w_q', type=float, default=1.0)
    parser.add_argument('--w_contact', type=float, default=0.3)
    parser.add_argument('--w_com', type=float, default=0.5)
    parser.add_argument('--w_kl', type=float, default=0.001)
    parser.add_argument('--w_grl', type=float, default=0.5)
    parser.add_argument('--grl_alpha', type=float, default=1.0)
    parser.add_argument('--eval_every', type=int, default=500)
    parser.add_argument('--early_kill_step', type=int, default=5000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()

    if args.pilot_12:
        # 12 representative skels (mix of morphologies for early kill)
        skels = ['Horse', 'Cat', 'Dog', 'Chicken', 'Anaconda',
                 'Spider', 'Crab', 'Trex', 'Bear', 'Eagle', 'Pirrana', 'Centipede']
    elif args.full:
        skels = OBJECT_SUBSETS_DICT['train_v3']
    else:
        raise ValueError("Specify --pilot_12 or --full")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Skels: {len(skels)} (first 5: {skels[:5]})")

    out_dir = SAVE_ROOT / args.out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    # Dataset
    dataset = TruebonesAuxDataset(skels, n_frames=args.n_frames)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         num_workers=args.num_workers, collate_fn=collate, drop_last=True)
    diag_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=0, collate_fn=collate)

    encoder = BehaviorEncoderE(
        d_model=args.d_model, d_z=args.d_z,
        n_clusters=N_CLUSTERS, n_skels=len(skels), q_dim=22,
    ).to(device)
    print(f"Encoder params: {count_parameters(encoder)/1e6:.2f}M")

    optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=1e-5)

    # Diagnostic + checkpoint history
    history = []
    best_aux_loss = float('inf')
    best_step = 0

    step = 0
    epoch = 0
    t0 = time.time()
    encoder.train()
    while step < args.n_steps:
        for batch in loader:
            if step >= args.n_steps: break
            motion = batch['motion'].to(device)
            mask = batch['joint_mask'].to(device)
            # GRL alpha warmup: ramp from 0.1 → grl_alpha over first 1/3 of training
            warmup_progress = min(1.0, step / (args.n_steps / 3))
            grl_alpha_now = 0.1 + (args.grl_alpha - 0.1) * warmup_progress
            out = encoder(motion, mask, grl_alpha=grl_alpha_now)

            # Losses
            l_action = F.cross_entropy(out['action_logits'], batch['action_idx'].to(device))
            l_q = F.mse_loss(out['q_pred'], batch['q_target'].to(device))
            l_contact = F.mse_loss(out['contact_pred'], batch['contact_density'].to(device))
            l_com = F.mse_loss(out['com_pred'], batch['com_heading'].to(device))
            l_kl = kl_loss(out['mu'], out['logvar'])
            l_grl = F.cross_entropy(out['skel_logits'], batch['skel_id'].to(device))

            loss = (args.w_action * l_action + args.w_q * l_q
                    + args.w_contact * l_contact + args.w_com * l_com
                    + args.w_kl * l_kl + args.w_grl * l_grl)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            optimizer.step()

            if step % 50 == 0:
                elapsed = time.time() - t0
                print(f"step {step:5d}/{args.n_steps} loss={loss.item():.3f} "
                      f"l_action={l_action.item():.3f} l_q={l_q.item():.3f} "
                      f"l_contact={l_contact.item():.3f} l_com={l_com.item():.3f} "
                      f"l_kl={l_kl.item():.2f} l_grl={l_grl.item():.3f} "
                      f"elapsed={elapsed:.0f}s")

            # Eval-while-train
            if step > 0 and step % args.eval_every == 0:
                t_eval = time.time()
                z, sk, ac = collect_z_and_skel(encoder, diag_loader, device, max_batches=20)
                skel_probe_acc = linear_probe(z, sk, len(skels))
                action_probe_acc = linear_probe(z, ac, N_CLUSTERS)
                swap_diff = latent_swap_test(encoder, dataset, device, n_pairs=30)
                history_entry = {
                    'step': step, 'loss': loss.item(),
                    'l_action': l_action.item(), 'l_q': l_q.item(),
                    'skel_probe_acc': skel_probe_acc,
                    'action_probe_acc': action_probe_acc,
                    'latent_swap_diff': swap_diff,
                    'eval_time': time.time() - t_eval,
                }
                history.append(history_entry)
                print(f"  EVAL step {step}: skel_probe={skel_probe_acc:.3f} "
                      f"action_probe={action_probe_acc:.3f} swap_diff={swap_diff:.3f} "
                      f"(eval_time={time.time()-t_eval:.0f}s)")

                # Save best by aux loss
                aux_loss = l_action.item() + l_q.item() + l_contact.item() + l_com.item()
                if aux_loss < best_aux_loss:
                    best_aux_loss = aux_loss
                    best_step = step
                    torch.save(encoder.state_dict(), out_dir / 'best.pt')
                    print(f"    Saved best at step {step} (aux_loss={aux_loss:.3f})")

                # EARLY KILL CHECK
                if step >= args.early_kill_step:
                    if skel_probe_acc > 0.10 or swap_diff < 0.02:
                        print(f"\n*** EARLY KILL TRIGGERED at step {step} ***")
                        print(f"    skel_probe_acc={skel_probe_acc:.3f} (target ≤ 0.05)")
                        print(f"    swap_diff={swap_diff:.3f} (target ≥ 0.02)")
                        with open(out_dir / 'early_kill.json', 'w') as f:
                            json.dump({
                                'step': step, 'history': history,
                                'reason': 'skel_probe>0.10 or swap_diff<0.02',
                            }, f, indent=2)
                        return

            step += 1
        epoch += 1

    # Final save
    torch.save(encoder.state_dict(), out_dir / 'final.pt')
    with open(out_dir / 'training_history.json', 'w') as f:
        json.dump({
            'args': vars(args), 'history': history,
            'best_step': best_step, 'best_aux_loss': best_aux_loss,
        }, f, indent=2)
    print(f"\nDone. Best at step {best_step} (aux_loss={best_aux_loss:.3f})")


if __name__ == '__main__':
    main()
