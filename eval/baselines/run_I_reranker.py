"""I — Discriminative reranker for cross-skel motion candidate selection.

Per FINAL_PROPOSAL V4: pairwise ranker on (source_motion, candidate_motion,
src_skel, tgt_skel). Trained on pseudo-labels (positive = candidate.action
matches source.action). Used downstream to select best candidate from
M3 Phase A union pool (I-5 ∪ k_retrieve ∪ E-sample).

Features (per src-cand pair):
  - src_Q (22d)
  - cand_Q (22d)
  - |src_Q - cand_Q| (22d)
  - cluster_match (1d)  (cand.cluster == cluster_pred from I-5 classifier)
  - q_cosine (1d)       (cosine sim of normalized Q sigs)
  - q_l2 (1d)           (L2 distance of Q sigs)

Total feature dim = 22*3 + 3 = 69.

Architecture: 3-layer MLP (69 → 128 → 64 → 1).
Loss: pairwise margin ranking (positive > negative + margin).

Training data: for each v5 query, generate pairs (positive_action_cand, negative_action_cand).

Usage:
  python -m eval.baselines.run_I_reranker --train --out_tag I_reranker_v1
  python -m eval.baselines.run_I_reranker --infer --folds 42 43 \
      --base_phase_a save/m3/m3_rerank_v1 --out_tag I_reranker_v1
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loaders.truebones.truebones_utils.param_utils import OBJECT_SUBSETS_DICT, DATASET_DIR
from eval.benchmark_v3.action_taxonomy import (
    ACTION_CLUSTERS, parse_action_from_filename, action_to_cluster,
)
from eval.baselines.run_i5_action_classifier_v3 import featurize_q, train_classifier, CLUSTERS

DATA_ROOT = Path(DATASET_DIR)
MOTION_DIR = DATA_ROOT / 'motions'
Q_CACHE_PATH = PROJECT_ROOT / 'idea-stage/quotient_cache.npz'
QUERIES_V5_DIR = PROJECT_ROOT / 'eval/benchmark_v3/queries_v5'
CLIP_INDEX_PATH = PROJECT_ROOT / 'eval/benchmark_v3/clip_index.json'
SAVE_ROOT = PROJECT_ROOT / 'save/I_reranker'


# ---------------- Model ----------------

class RerankerMLP(nn.Module):
    def __init__(self, feat_dim=69, hidden=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def featurize_pair(src_q22: np.ndarray, cand_q22: np.ndarray,
                   cluster_match: float, q_cosine: float, q_l2: float,
                   action_match: float = 0.0) -> np.ndarray:
    """Return 70-d feature vector for a (src, candidate) pair."""
    diff = np.abs(src_q22 - cand_q22)
    return np.concatenate([
        src_q22, cand_q22, diff,
        np.array([cluster_match, q_cosine, q_l2, action_match], dtype=np.float32),
    ]).astype(np.float32)


# ---------------- Pair construction ----------------

def build_train_pairs(qc, target_pool, fname_to_q22, fname_to_cluster,
                       train_skels, max_per_query=10, seed=42, hard_only=False,
                       mix_easy_hard=True):
    """Generate (src, positive_target, negative_target) triples for pairwise ranking.

    HARD PAIRS (default): both pos and neg in SAME cluster as src. Pos has exact
    same action as src; neg has different action. This forces the reranker to
    use Q-features rather than cluster_match (which would be 1.0 for both).

    Easy pairs: pos in same cluster, neg in different cluster (cluster_match
    discriminates trivially).
    """
    rng = np.random.RandomState(seed)
    train_skels_set = set(train_skels)

    # Build (skel, cluster) -> [(fname, action)] for finer slicing
    by_skel_cluster = defaultdict(list)
    for m in qc['meta']:
        action = parse_action_from_filename(m['fname'])
        cluster = action_to_cluster(action)
        if cluster is None: continue
        by_skel_cluster[(m['skeleton'], cluster)].append((m['fname'], action))

    train_meta = [(i, m) for i, m in enumerate(qc['meta']) if m['skeleton'] in train_skels_set]
    print(f"Building train pairs from {len(train_meta)} train clips ({'hard' if hard_only else 'easy'})...")

    pairs = []
    for src_idx, src_m in train_meta:
        src_skel = src_m['skeleton']
        src_action = parse_action_from_filename(src_m['fname'])
        src_cluster = action_to_cluster(src_action)
        if src_cluster is None: continue
        src_q22 = fname_to_q22[src_m['fname']]
        candidate_target_skels = [s for s in train_skels if s != src_skel]
        if not candidate_target_skels: continue
        for k in range(max_per_query):
            tgt_skel = rng.choice(candidate_target_skels)
            # Mix easy/hard: half easy (cluster discriminator), half hard (within-cluster)
            use_hard = (k % 2 == 1) if mix_easy_hard else hard_only
            if use_hard:
                # HARD: both pos and neg in same cluster on tgt_skel; differ by action
                same_cluster_clips = by_skel_cluster.get((tgt_skel, src_cluster), [])
                if len(same_cluster_clips) < 2: continue
                pos_candidates = [(f, a) for f, a in same_cluster_clips if a == src_action]
                neg_candidates = [(f, a) for f, a in same_cluster_clips if a != src_action]
                if not pos_candidates or not neg_candidates: continue
                pos_fname, pos_action = pos_candidates[rng.randint(len(pos_candidates))]
                neg_fname, neg_action = neg_candidates[rng.randint(len(neg_candidates))]
                cm_pos = 1.0; cm_neg = 1.0  # both in cluster_pred bucket
                am_pos = 1.0; am_neg = 0.0
            else:
                pos_pool = target_pool.get((tgt_skel, src_cluster), [])
                if not pos_pool: continue
                pos_fname = rng.choice(pos_pool)
                neg_clusters = [c for c in CLUSTERS if c != src_cluster]
                neg_fname = None
                rng.shuffle(neg_clusters)
                for nc in neg_clusters:
                    neg_pool = target_pool.get((tgt_skel, nc), [])
                    if neg_pool:
                        neg_fname = rng.choice(neg_pool); break
                if neg_fname is None: continue
                cm_pos = 1.0; cm_neg = 0.0
                # action_match in easy: pos.action == src.action only if pos was sampled with that action
                pos_action_in_pool = parse_action_from_filename(pos_fname)
                neg_action_in_pool = parse_action_from_filename(neg_fname)
                am_pos = 1.0 if pos_action_in_pool == src_action else 0.0
                am_neg = 1.0 if neg_action_in_pool == src_action else 0.0

            if pos_fname not in fname_to_q22 or neg_fname not in fname_to_q22:
                continue
            pairs.append({
                'src_q22': src_q22, 'pos_q22': fname_to_q22[pos_fname],
                'neg_q22': fname_to_q22[neg_fname],
                'cluster_match_pos': cm_pos, 'cluster_match_neg': cm_neg,
                'action_match_pos': am_pos, 'action_match_neg': am_neg,
                'src_fname': src_m['fname'], 'pos_fname': pos_fname, 'neg_fname': neg_fname,
            })
    print(f"Generated {len(pairs)} pairs.")
    return pairs


class PairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        p = self.pairs[i]
        src_q22 = p['src_q22']
        cm_pos = p['cluster_match_pos']
        cm_neg = p['cluster_match_neg']
        am_pos = p.get('action_match_pos', 0.0)
        am_neg = p.get('action_match_neg', 0.0)
        sn = src_q22 / (np.linalg.norm(src_q22) + 1e-9)
        pn = p['pos_q22'] / (np.linalg.norm(p['pos_q22']) + 1e-9)
        nn_ = p['neg_q22'] / (np.linalg.norm(p['neg_q22']) + 1e-9)
        q_cos_pos = float(sn @ pn); q_l2_pos = float(np.linalg.norm(src_q22 - p['pos_q22']))
        q_cos_neg = float(sn @ nn_); q_l2_neg = float(np.linalg.norm(src_q22 - p['neg_q22']))
        feat_pos = featurize_pair(src_q22, p['pos_q22'], cm_pos, q_cos_pos, q_l2_pos, am_pos)
        feat_neg = featurize_pair(src_q22, p['neg_q22'], cm_neg, q_cos_neg, q_l2_neg, am_neg)
        return torch.from_numpy(feat_pos), torch.from_numpy(feat_neg)


# ---------------- Training ----------------

def train_reranker(out_tag, n_steps=2000, lr=1e-3, batch_size=64, margin=1.0, device='cuda'):
    out_dir = SAVE_ROOT / out_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Q cache...")
    qc = np.load(Q_CACHE_PATH, allow_pickle=True)
    fname_to_q22 = {}
    for i, m in enumerate(qc['meta']):
        fname_to_q22[m['fname']] = featurize_q(qc['com_path'][i], qc['heading_vel'][i],
                                                qc['contact_sched'][i], qc['cadence'][i],
                                                qc['limb_usage'][i])

    fname_to_cluster = {}
    for m in qc['meta']:
        action = parse_action_from_filename(m['fname'])
        fname_to_cluster[m['fname']] = action_to_cluster(action)

    # Target pool from clip_index
    cidx = json.load(open(CLIP_INDEX_PATH))
    target_pool = defaultdict(list)
    for skel, clusters in cidx['index'].items():
        for cluster, clips in clusters.items():
            for clip in clips:
                target_pool[(skel, cluster)].append(clip['fname'])

    train_skels = OBJECT_SUBSETS_DICT['train_v3']
    pairs = build_train_pairs(qc, target_pool, fname_to_q22, fname_to_cluster, train_skels,
                              max_per_query=10)
    if not pairs:
        raise RuntimeError("No training pairs generated")

    dataset = PairDataset(pairs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = RerankerMLP(feat_dim=70, hidden=128).to(device)
    print(f"Reranker params: {sum(p.numel() for p in model.parameters())/1e3:.1f}K")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    step = 0
    history = []
    t0 = time.time()
    while step < n_steps:
        for feat_pos, feat_neg in loader:
            if step >= n_steps: break
            feat_pos = feat_pos.to(device); feat_neg = feat_neg.to(device)
            score_pos = model(feat_pos)
            score_neg = model(feat_neg)
            # Pairwise margin ranking: pos > neg + margin
            loss = F.relu(margin - (score_pos - score_neg)).mean()
            # Also add contrastive accuracy
            acc = (score_pos > score_neg).float().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                elapsed = time.time() - t0
                history.append({'step': step, 'loss': loss.item(), 'acc': acc.item()})
                print(f"  step {step:4d}/{n_steps} loss={loss.item():.3f} "
                      f"acc={acc.item():.3f} elapsed={elapsed:.0f}s")
            step += 1

    torch.save(model.state_dict(), out_dir / 'reranker.pt')
    with open(out_dir / 'training.json', 'w') as f:
        json.dump({'n_steps': n_steps, 'history': history,
                   'final_loss': history[-1]['loss'], 'final_acc': history[-1]['acc']}, f, indent=2)
    print(f"\nSaved reranker to {out_dir}/reranker.pt")
    return out_dir


# ---------------- Inference ----------------

def infer_rerank(out_tag, folds, base_phase_a_dir, topk_q=10, device='cuda'):
    """For each query in v5 fold, build candidate pool (I-5 + k_retrieve top-K),
    score with reranker, pick top-1. Save as query_NNNN.npy."""
    out_dir_base = SAVE_ROOT / out_tag
    model_path = out_dir_base / 'reranker.pt'
    if not model_path.exists():
        raise RuntimeError(f"Model not found: {model_path}. Train first.")

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = RerankerMLP(feat_dim=70, hidden=128).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("Loading Q cache and I-5 classifier...")
    qc = np.load(Q_CACHE_PATH, allow_pickle=True)
    fname_to_q22 = {}
    for i, m in enumerate(qc['meta']):
        fname_to_q22[m['fname']] = featurize_q(qc['com_path'][i], qc['heading_vel'][i],
                                                qc['contact_sched'][i], qc['cadence'][i],
                                                qc['limb_usage'][i])
    train_skels = set(OBJECT_SUBSETS_DICT['train_v3'])
    clf = train_classifier(qc, train_skels)
    fname_to_idx = {m['fname']: i for i, m in enumerate(qc['meta'])}
    cidx = json.load(open(CLIP_INDEX_PATH))

    base_phase_a_root = Path(base_phase_a_dir)
    if not base_phase_a_root.is_absolute():
        base_phase_a_root = PROJECT_ROOT / base_phase_a_root

    for fold in folds:
        manifest = json.load(open(QUERIES_V5_DIR / f'fold_{fold}/manifest.json'))
        out_dir = out_dir_base / f'fold_{fold}'
        out_dir.mkdir(parents=True, exist_ok=True)

        n_done = n_failed = 0
        per_query = []
        t0 = time.time()
        for i, q in enumerate(manifest['queries']):
            qid = q['query_id']
            skel_b = q['skel_b']
            src_fname = q['src_fname']
            try:
                # Source Q
                src_q22 = fname_to_q22[src_fname]
                src_n = src_q22 / (np.linalg.norm(src_q22) + 1e-9)
                # Cluster_pred from I-5
                pred_cluster_idx = int(clf.predict(src_q22[None, :])[0])
                pred_cluster = CLUSTERS[pred_cluster_idx]

                # Build candidate pool: cluster_pred + top-K Q-sim across full library
                all_skel_b_clips = []
                for cluster, clips in cidx['index'][skel_b].items():
                    for c in clips:
                        all_skel_b_clips.append({**c, 'cluster': cluster})

                pool = {}
                # Cluster pool
                for c in all_skel_b_clips:
                    if c['cluster'] == pred_cluster:
                        pool[c['fname']] = c
                # Q pool
                q_ranked = []
                for c in all_skel_b_clips:
                    if c['fname'] not in fname_to_q22: continue
                    cn = fname_to_q22[c['fname']] / (np.linalg.norm(fname_to_q22[c['fname']]) + 1e-9)
                    qsim = float(src_n @ cn)
                    q_ranked.append((qsim, c))
                q_ranked.sort(key=lambda x: -x[0])
                for _, c in q_ranked[:topk_q]:
                    pool[c['fname']] = c

                if not pool:
                    pool = {c['fname']: c for c in all_skel_b_clips}

                # Score every candidate with reranker
                feats = []; cand_records = []
                for fname, c in pool.items():
                    if fname not in fname_to_q22: continue
                    cand_q22 = fname_to_q22[fname]
                    cm = 1.0 if c['cluster'] == pred_cluster else 0.0
                    cn = cand_q22 / (np.linalg.norm(cand_q22) + 1e-9)
                    qcos = float(src_n @ cn)
                    ql2 = float(np.linalg.norm(src_q22 - cand_q22))
                    am = 1.0 if c.get('action') == q.get('src_action') else 0.0
                    feat = featurize_pair(src_q22, cand_q22, cm, qcos, ql2, am)
                    feats.append(feat)
                    cand_records.append((fname, c, qcos, cm, am))

                if not feats:
                    raise RuntimeError("Empty pool")

                feats_t = torch.from_numpy(np.stack(feats)).to(device)
                with torch.no_grad():
                    scores = model(feats_t).cpu().numpy()
                best_i = int(np.argmax(scores))
                best_fname, best_c, best_qcos, best_cm, best_am = cand_records[best_i]
                motion = np.load(MOTION_DIR / best_fname).astype(np.float32)
                np.save(out_dir / f'query_{qid:04d}.npy', motion)
                per_query.append({
                    'query_id': qid, 'status': 'ok',
                    'picked_fname': best_fname, 'picked_action': best_c['action'],
                    'picked_cluster': best_c['cluster'], 'pred_cluster': pred_cluster,
                    'best_score': float(scores[best_i]), 'q_cos': best_qcos,
                    'cluster_match': best_cm, 'action_match': best_am,
                    'pool_size': len(cand_records),
                })
                n_done += 1
            except Exception as e:
                import traceback
                tb = traceback.format_exc(limit=2)
                print(f"  q{qid} FAILED: {e}\n{tb}")
                per_query.append({'query_id': qid, 'status': 'failed', 'error': str(e)})
                n_failed += 1

            if (i + 1) % 50 == 0 or i == 0:
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (len(manifest['queries']) - i - 1)
                print(f"  fold {fold} [{i+1}/{len(manifest['queries'])}] "
                      f"elapsed {elapsed:.0f}s, ETA {eta:.0f}s, ok={n_done}")

        with open(out_dir / 'metrics.json', 'w') as f:
            json.dump({'method': out_tag, 'fold': fold, 'n_done': n_done,
                       'n_failed': n_failed, 'per_query': per_query}, f, indent=2)
        print(f"\nFold {fold}: {n_done} ok, {n_failed} failed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--infer', action='store_true')
    parser.add_argument('--out_tag', type=str, default='I_reranker_v1')
    parser.add_argument('--n_steps', type=int, default=2000)
    parser.add_argument('--folds', nargs='+', type=int, default=[42, 43])
    parser.add_argument('--base_phase_a', type=str, default='save/m3/m3_rerank_v1')
    parser.add_argument('--topk_q', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    if args.train:
        train_reranker(args.out_tag, n_steps=args.n_steps, device=args.device)
    if args.infer:
        infer_rerank(args.out_tag, args.folds, args.base_phase_a,
                     topk_q=args.topk_q, device=args.device)


if __name__ == '__main__':
    main()
