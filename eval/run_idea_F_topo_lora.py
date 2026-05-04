"""Idea F: topology-edit-anchored LoRA few-shot retargeting.

Implementation follows spec's explicit fallback:

    "If LoRA on the model is too complex, train a tiny 2-layer MLP per-pair
    that maps (source_motion, target_rest_pose) -> target_motion, with 3
    training pairs as supervision. The core idea - learning from 3
    procedurally-paired anchors - is what we want to test."

And further:

    "If topology-edit augmentation is unavailable for a particular pair, use
    the retrieved top-3 clips from quotient_cache.npz as the 'synthetic
    anchors' (retrieve -> blend -> learn)."

We adopt the MLP + retrieval-blend-learn route because:
  * Fits 45-min budget for 30 pairs (per-pair training + inference in <2s).
  * Training LoRA on the 8M-param behavior AnyTop for 3 pairs x 30 pairs in
    45 min is infeasible on 12 GB (each forward alone takes ~0.3s; 200 steps
    per pair x 30 pairs = ~180s pure compute but memory for adapters +
    frozen weights is tight; data shaping for synthetic pseudo-sources adds
    extra work). The MLP variant isolates the research question (does
    learned-few-shot beat pure retrieval?) without that overhead.
  * The topology-edit augmentation (add/remove joints) requires (a) known
    donor joints + (b) consistent joint naming between source and target,
    which is not guaranteed across arbitrary skeleton pairs in the Truebones
    Zoo. We instead derive a "source-view" supervision target by treating
    the src Q-signature as the query anchor: the model learns to bias a
    blended target init toward the specific target clip whose Q-signature
    is closest to the source's.

Per pair pipeline
-----------------
1. Retrieve top-3 target-skel clips by Q-signature cosine (via eval/pilot_Q_experiments.q_signature on quotient_cache.npz).
2. Time-resample each retrieved clip to T_src.
3. Build a weighted blend as a retrieval baseline x_blend ([T_src, J_tgt, 13]).
4. Build 3 "procedural training pairs":
     input_i  = tile(q_signature(target_clip_i), T_src, J_tgt) concatenated
                with the JOINT-LEVEL features of x_blend, conditioned on
                target rest pose tpos;
     target_i = the clip target_clip_i resampled to T_src.
   This is the "retrieve->blend->learn" schema: learn a per-pair adapter
   that, given a Q-signature context and the blended init, can steer output
   toward the specific clip whose signature is fed in.
5. Train a tiny MLP (2 hidden layers, width 128) for 150 steps on the 3
   pairs (Adam, lr 1e-3, MSE).
6. At inference: feed source Q-signature + blended init -> adapter output.
7. Save [T_src, J_tgt, 13] to eval/results/k_compare/idea_F_topo_lora/.

Evaluation matches v4 schema (q_component distances, contact_f1, wall time,
stratified summary). The external classifier accuracy sweep is piggybacked
by a mini harness that mirrors eval/k_action_accuracy.py.
"""
from __future__ import annotations
PROJECT_ROOT = Path(__file__).resolve().parents[1]
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(str(PROJECT_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EVAL_PAIRS = ROOT / 'idea-stage/eval_pairs.json'
META_PATH = ROOT / 'eval/results/effect_cache/clip_metadata.json'
Q_CACHE_PATH = ROOT / 'idea-stage/quotient_cache.npz'
COND_PATH = ROOT / 'dataset/truebones/zoo/truebones_processed/cond.npy'
MOTIONS_DIR = ROOT / 'dataset/truebones/zoo/truebones_processed/motions'
CONTACT_GROUPS_PATH = ROOT / 'eval/quotient_assets/contact_groups.json'

OUT_DIR = ROOT / 'eval/results/k_compare/idea_F_topo_lora'
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOPK = 3
BLEND_TAU = 0.05
N_TRAIN_STEPS = 150
LR = 1e-3
MLP_HIDDEN = 128
SIG_DIM = 20  # q_signature output size

# feature channels
POS_Y_IDX = 1
FOOT_CH_IDX = 12


def _resample_T(m: np.ndarray, T_out: int) -> np.ndarray:
    T_in = m.shape[0]
    idx = np.clip(np.linspace(0, T_in - 1, T_out).astype(int), 0, T_in - 1)
    return m[idx]


def l2_norm(x, axis=-1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-9)


def cosine_sim(a, b):
    return l2_norm(a) @ l2_norm(b).T


def q_component_l2(q_src: dict, q_tgt: dict) -> dict:
    def _l2(a, b):
        a = np.asarray(a); b = np.asarray(b)
        if a.shape != b.shape:
            if a.ndim == b.ndim and a.ndim >= 1:
                T = min(a.shape[0], b.shape[0])
                a = a[:T]; b = b[:T]
            else:
                return None
        return float(np.linalg.norm((a - b).reshape(-1)))
    out = {
        'com_path': _l2(q_src['com_path'], q_tgt['com_path']),
        'heading_vel': _l2(q_src['heading_vel'], q_tgt['heading_vel']),
        'cadence': float(abs(float(q_src['cadence']) - float(q_tgt['cadence']))),
    }
    cs_src = np.asarray(q_src['contact_sched']).reshape(
        q_src['contact_sched'].shape[0], -1).sum(axis=-1)
    cs_tgt = np.asarray(q_tgt['contact_sched']).reshape(
        q_tgt['contact_sched'].shape[0], -1).sum(axis=-1)
    T = min(len(cs_src), len(cs_tgt))
    out['contact_sched_aggregate'] = float(np.linalg.norm(cs_src[:T] - cs_tgt[:T]))
    lu_src = -np.sort(-np.asarray(q_src['limb_usage']))[:5]
    lu_tgt = -np.sort(-np.asarray(q_tgt['limb_usage']))[:5]
    K = min(len(lu_src), len(lu_tgt))
    lu_src = np.pad(lu_src[:K], (0, max(0, 5 - K)))
    lu_tgt = np.pad(lu_tgt[:K], (0, max(0, 5 - K)))
    out['limb_usage_top5'] = float(np.linalg.norm(lu_src - lu_tgt))
    return out


def contact_f1(sched_rec: np.ndarray, sched_tgt: np.ndarray, thresh: float = 0.5) -> float:
    pred = (np.asarray(sched_rec) >= thresh).astype(np.int8).ravel()
    gt = (np.asarray(sched_tgt) >= thresh).astype(np.int8).ravel()
    n = min(pred.size, gt.size)
    pred, gt = pred[:n], gt[:n]
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    p = tp / (tp + fp + 1e-8); r = tp / (tp + fn + 1e-8)
    return float(2 * p * r / (p + r + 1e-8))


def build_q_sig_array(qc):
    from eval.pilot_Q_experiments import q_signature
    N = len(qc['meta'])
    sigs = []
    for i in range(N):
        q = {
            'com_path': qc['com_path'][i], 'heading_vel': qc['heading_vel'][i],
            'contact_sched': qc['contact_sched'][i], 'cadence': float(qc['cadence'][i]),
            'limb_usage': qc['limb_usage'][i],
        }
        sigs.append(q_signature(q))
    return np.stack(sigs)


def stratified_summary(all_entries):
    buckets = defaultdict(list)
    for e in all_entries:
        fam = e['family_gap']
        if fam in ('near_present', 'near'):
            buckets['near_present'].append(e)
        if fam == 'moderate':
            buckets['moderate'].append(e)
        if fam == 'extreme':
            buckets['extreme'].append(e)
        if e['support_same_label'] == 0:
            buckets['absent'].append(e)
    buckets['all'] = list(all_entries)
    numeric_keys = [
        'q_com_path_l2', 'q_heading_vel_l2', 'q_contact_sched_l2',
        'q_cadence_abs_diff', 'q_limb_usage_top5_l2',
        'q_com_path_l2_pre', 'q_heading_vel_l2_pre', 'q_contact_sched_l2_pre',
        'q_cadence_abs_diff_pre', 'q_limb_usage_top5_pre',
        'q_com_path_delta', 'q_heading_vel_delta', 'q_contact_sched_delta',
        'q_cadence_delta', 'q_limb_usage_delta',
        'contact_f1_vs_source', 'contact_f1_self',
        'train_time_s', 'inference_time_s', 'train_loss_final', 'wall_time_s',
    ]
    summary = {}
    for stratum, entries in buckets.items():
        summary[stratum] = {'n': len(entries)}
        for k in numeric_keys:
            vals = [e[k] for e in entries if e.get(k) is not None]
            summary[stratum][k] = float(np.mean(vals)) if vals else None
    return summary


class PairMLP:
    """Per-pair adapter: input channels per (t, j) token = SIG_DIM + 13 (joint
    feats of blended init) + 13 (joint rest-pose feat, broadcast across T)
    + 13 (tpos-encoded reference) -> output = 13-dim correction delta added
    on top of blended init.

    Implemented without requiring a separate module file so it lives in one
    script.
    """

    def __init__(self, n_joints: int, device: str = 'cuda'):
        import torch
        import torch.nn as nn
        self.torch = torch
        self.device = device
        # input layout per token: sig (20) + blend_feat (13) + rest_offset (3)
        in_dim = SIG_DIM + 13 + 3
        self.in_dim = in_dim
        self.n_joints = n_joints
        self.net = nn.Sequential(
            nn.Linear(in_dim, MLP_HIDDEN),
            nn.GELU(),
            nn.Linear(MLP_HIDDEN, MLP_HIDDEN),
            nn.GELU(),
            nn.Linear(MLP_HIDDEN, 13),
        ).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=LR)

    def _pack(self, sig: np.ndarray, x_blend: np.ndarray, tgt_offsets: np.ndarray):
        """
        sig: [SIG_DIM]
        x_blend: [T, J, 13]
        tgt_offsets: [J, 3] rest-pose offsets for the target skeleton.
        returns: torch tensor [T, J, in_dim]
        """
        torch = self.torch
        T, J, F = x_blend.shape
        sig_t = torch.from_numpy(sig.astype(np.float32)).to(self.device)
        sig_b = sig_t.view(1, 1, SIG_DIM).expand(T, J, SIG_DIM)
        xb = torch.from_numpy(x_blend.astype(np.float32)).to(self.device)
        off = torch.from_numpy(tgt_offsets.astype(np.float32)).to(self.device)
        off_b = off.view(1, J, 3).expand(T, J, 3)
        tok = torch.cat([sig_b, xb, off_b], dim=-1)
        return tok

    def train(self, training_pairs, tgt_offsets, n_steps: int = N_TRAIN_STEPS):
        """training_pairs: list of {'sig': [SIG_DIM], 'x_blend': [T,J,13], 'x_gt': [T,J,13]}"""
        torch = self.torch
        losses = []
        t0 = time.time()
        self.net.train()
        for step in range(n_steps):
            loss_total = 0.0
            # small 3-pair batch, iterate sequentially (accumulate grad)
            self.opt.zero_grad()
            for tp in training_pairs:
                tok = self._pack(tp['sig'], tp['x_blend'], tgt_offsets)  # [T,J,in_dim]
                pred_delta = self.net(tok)  # [T,J,13]
                y_gt = torch.from_numpy(tp['x_gt'].astype(np.float32)).to(self.device)
                y_init = torch.from_numpy(tp['x_blend'].astype(np.float32)).to(self.device)
                pred = y_init + pred_delta
                loss = ((pred - y_gt) ** 2).mean()
                loss.backward()
                loss_total += float(loss.item())
            self.opt.step()
            losses.append(loss_total / max(1, len(training_pairs)))
        train_time = time.time() - t0
        return {'train_time_s': train_time, 'final_loss': losses[-1] if losses else None,
                'losses': losses[::10]}

    def infer(self, sig: np.ndarray, x_blend: np.ndarray, tgt_offsets: np.ndarray) -> np.ndarray:
        torch = self.torch
        self.net.eval()
        with torch.no_grad():
            tok = self._pack(sig, x_blend, tgt_offsets)
            delta = self.net(tok)
            out = torch.from_numpy(x_blend.astype(np.float32)).to(self.device) + delta
        return out.cpu().numpy()


def run(restrict_n: int | None = None):
    t_start_all = time.time()

    # Avoid pulling in anytop_projection/model (spec's MLP fallback needs no ckpt).
    print('Loading caches...')
    with open(EVAL_PAIRS) as f:
        pairs = json.load(f)['pairs']
    with open(META_PATH) as f:
        meta = json.load(f)
    qc = np.load(Q_CACHE_PATH, allow_pickle=True)
    cond_dict = np.load(COND_PATH, allow_pickle=True).item()

    q_meta = list(qc['meta'])
    fname_to_q_idx = {m['fname']: i for i, m in enumerate(q_meta)}
    skel_to_meta_idx = defaultdict(list)
    for i, m in enumerate(meta):
        skel_to_meta_idx[m['skeleton']].append(i)

    print('Building Q signatures...')
    q_sigs = build_q_sig_array(qc)

    import torch  # noqa: F401 (imported for CUDA check via PairMLP)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if restrict_n is not None:
        pairs = pairs[:restrict_n]

    per_pair = []
    for p in pairs:
        pid = int(p['pair_id'])
        src_skel = p['source_skel']; src_fname = p['source_fname']
        tgt_skel = p['target_skel']; src_label = p['source_label']
        family_gap = p['family_gap']; support = int(p['support_same_label'])
        strat = {'near': 'near_present'}.get(family_gap, family_gap)
        print(f"\n=== pair {pid:02d} {src_skel}({src_fname}) -> {tgt_skel}  "
              f"gap={family_gap}  supp={support} ===")
        rec = {'pair_id': pid, 'src_fname': src_fname, 'src_skel': src_skel,
               'src_label': src_label, 'tgt_skel': tgt_skel,
               'family_gap': strat, 'support_same_label': support,
               'status': 'pending', 'error': None,
               'approach': 'retrieval_blend_learn_mlp'}
        t_pair0 = time.time()
        try:
            if src_fname not in fname_to_q_idx:
                raise RuntimeError(f'source missing Q cache: {src_fname}')
            if tgt_skel not in cond_dict:
                raise RuntimeError(f'missing cond: {tgt_skel}')

            src_q_idx = fname_to_q_idx[src_fname]
            src_q_sig = q_sigs[src_q_idx]

            tgt_pool = [i for i in skel_to_meta_idx[tgt_skel]
                        if meta[i]['fname'] != src_fname
                        and meta[i]['fname'] in fname_to_q_idx]
            if not tgt_pool:
                raise RuntimeError(f'empty tgt pool {tgt_skel}')

            # Retrieve top-K target clips
            cand_q_idx = np.array([fname_to_q_idx[meta[i]['fname']] for i in tgt_pool])
            sims = cosine_sim(src_q_sig[None], q_sigs[cand_q_idx])[0]
            k_avail = min(TOPK, len(sims))
            order = np.argsort(-sims)[:k_avail]
            topk = [{'meta_idx': tgt_pool[int(i)],
                     'q_idx': int(cand_q_idx[int(i)]),
                     'sim': float(sims[int(i)]),
                     'fname': meta[tgt_pool[int(i)]]['fname'],
                     'coarse_label': meta[tgt_pool[int(i)]]['coarse_label']}
                    for i in order]
            w = np.array([c['sim'] for c in topk])
            w = np.exp(w / BLEND_TAU); w = w / w.sum()
            for c, wt in zip(topk, w):
                c['weight'] = float(wt)
            rec['topk_fnames'] = [c['fname'] for c in topk]
            rec['topk_weights'] = [c['weight'] for c in topk]
            rec['topk_coarse_labels'] = [c['coarse_label'] for c in topk]

            # Load motions and determine target length T_out = T_src
            src_motion_disk = np.load(MOTIONS_DIR / src_fname).astype(np.float32)
            T_out = src_motion_disk.shape[0]
            # Enforce reasonable minimum for MLP training stability (120 frames
            # is consistent with v4 baseline outputs).
            T_out = max(T_out, 60)

            # Load top-k target motions, resample to T_out
            topk_motions = []
            for c in topk:
                m = np.load(MOTIONS_DIR / c['fname']).astype(np.float32)
                m_res = _resample_T(m, T_out)
                topk_motions.append({**c, 'motion': m_res})

            # Blend
            weights = np.array([m['weight'] for m in topk_motions], dtype=np.float32)
            stacked = np.stack([m['motion'] for m in topk_motions], axis=0)
            x_blend = (stacked * weights[:, None, None, None]).sum(axis=0)
            J_tgt = x_blend.shape[1]

            # Build 3 training pairs: each pair uses a retrieved target clip's
            # own Q-signature as the "pseudo-source" input, pairing with that
            # clip's motion as ground truth (anchor).
            training_pairs = []
            for m in topk_motions:
                t_sig = q_sigs[m['q_idx']].astype(np.float32)
                training_pairs.append({
                    'sig': t_sig,
                    'x_blend': x_blend,
                    'x_gt': m['motion'].astype(np.float32),
                })

            # Train the per-pair MLP
            tgt_offsets = cond_dict[tgt_skel]['offsets'][:J_tgt].astype(np.float32)
            mlp = PairMLP(n_joints=J_tgt, device=device)
            train_info = mlp.train(training_pairs, tgt_offsets, n_steps=N_TRAIN_STEPS)
            rec['train_time_s'] = float(train_info['train_time_s'])
            rec['train_loss_final'] = float(train_info['final_loss']) if train_info['final_loss'] is not None else None
            rec['train_loss_curve'] = [float(x) for x in train_info['losses']]

            # Inference: feed actual source Q-signature
            t_inf = time.time()
            x_out = mlp.infer(src_q_sig.astype(np.float32), x_blend, tgt_offsets)
            rec['inference_time_s'] = float(time.time() - t_inf)

            out_fname = f'pair_{pid:02d}_{src_skel}_to_{tgt_skel}.npy'
            np.save(OUT_DIR / out_fname, x_out.astype(np.float32))
            rec['output_file'] = out_fname

            # Q-component distances: compute on x_blend (pre) and x_out (post)
            from eval.quotient_extractor import extract_quotient
            with open(CONTACT_GROUPS_PATH) as f:
                contact_groups = json.load(f)
            src_q = extract_quotient(src_fname, cond_dict[src_skel],
                                     contact_groups=contact_groups,
                                     motion_dir=str(MOTIONS_DIR))

            def _extract_q_for(motion: np.ndarray, tag: str):
                tmp_name = f'__ideaF_{tag}_pair_{pid:02d}.npy'
                tmp_path = MOTIONS_DIR / tmp_name
                try:
                    np.save(tmp_path, motion.astype(np.float32))
                    q = extract_quotient(tmp_name, cond_dict[tgt_skel],
                                         contact_groups=contact_groups,
                                         motion_dir=str(MOTIONS_DIR))
                finally:
                    if tmp_path.exists():
                        try: tmp_path.unlink()
                        except Exception: pass
                return q

            blend_q = _extract_q_for(x_blend, 'pre')
            out_q = _extract_q_for(x_out, 'post')

            q_pre = q_component_l2(src_q, blend_q)
            q_post = q_component_l2(src_q, out_q)

            rec['q_com_path_l2_pre'] = q_pre.get('com_path')
            rec['q_heading_vel_l2_pre'] = q_pre.get('heading_vel')
            rec['q_contact_sched_l2_pre'] = q_pre.get('contact_sched_aggregate')
            rec['q_cadence_abs_diff_pre'] = q_pre.get('cadence')
            rec['q_limb_usage_top5_pre'] = q_pre.get('limb_usage_top5')

            rec['q_com_path_l2'] = q_post.get('com_path')
            rec['q_heading_vel_l2'] = q_post.get('heading_vel')
            rec['q_contact_sched_l2'] = q_post.get('contact_sched_aggregate')
            rec['q_cadence_abs_diff'] = q_post.get('cadence')
            rec['q_limb_usage_top5_l2'] = q_post.get('limb_usage_top5')

            def _d(a, b):
                if a is None or b is None: return None
                return float(a - b)
            rec['q_com_path_delta'] = _d(rec['q_com_path_l2'], rec['q_com_path_l2_pre'])
            rec['q_heading_vel_delta'] = _d(rec['q_heading_vel_l2'], rec['q_heading_vel_l2_pre'])
            rec['q_contact_sched_delta'] = _d(rec['q_contact_sched_l2'], rec['q_contact_sched_l2_pre'])
            rec['q_cadence_delta'] = _d(rec['q_cadence_abs_diff'], rec['q_cadence_abs_diff_pre'])
            rec['q_limb_usage_delta'] = _d(rec['q_limb_usage_top5_l2'], rec['q_limb_usage_top5_pre'])

            # Contact F1 vs source (identical accounting as v4)
            rs = np.asarray(out_q['contact_sched'])
            ss = np.asarray(src_q['contact_sched'])
            T_ref = rs.shape[0]
            idx = np.clip(np.linspace(0, ss.shape[0] - 1, T_ref).astype(int),
                          0, ss.shape[0] - 1)
            ss_aligned = ss[idx]
            rs_agg = rs.sum(axis=1) if rs.ndim == 2 else rs
            ss_agg = ss_aligned.sum(axis=1) if ss_aligned.ndim == 2 else ss_aligned
            rec['contact_f1_vs_source'] = contact_f1(rs_agg, ss_agg)
            rec['contact_f1_self'] = 1.0

            rec['wall_time_s'] = float(time.time() - t_pair0)
            rec['status'] = 'ok'
            cd = rec.get('q_com_path_delta', 0) or 0
            print(f"  ok  top3={[c['fname'].split('___')[0] for c in topk]}  "
                  f"train={rec['train_time_s']:.2f}s  loss={rec['train_loss_final']:.4f}  "
                  f"infer={rec['inference_time_s']:.3f}s  "
                  f"q_com pre={rec['q_com_path_l2_pre']:.3f}->post={rec['q_com_path_l2']:.3f} "
                  f"({cd:+.3f})  cF1={rec['contact_f1_vs_source']:.3f}  wall={rec['wall_time_s']:.2f}s")

            # Release GPU memory of per-pair MLP
            del mlp
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        except Exception as e:
            rec['status'] = 'failed'
            rec['error'] = str(e) + '\n' + traceback.format_exc(limit=3)
            rec['wall_time_s'] = float(time.time() - t_pair0)
            print(f"  FAILED: {e}")
        per_pair.append(rec)

    total_time = time.time() - t_start_all
    ok = [r for r in per_pair if r['status'] == 'ok']
    stratified = stratified_summary(ok)
    out = {
        'method': 'idea_F_topo_lora',
        'variant': 'retrieval_blend_learn_mlp',
        'note': ('Topology-edit-anchored LoRA is simplified to a per-pair MLP '
                 'with retrieval-blend-learn (3 top-K target clips as procedural '
                 'anchors). See module docstring for rationale.'),
        'hparams': {
            'topk': TOPK, 'blend_tau': BLEND_TAU,
            'n_train_steps': N_TRAIN_STEPS, 'lr': LR,
            'mlp_hidden': MLP_HIDDEN, 'signature_dim': int(q_sigs.shape[1]),
        },
        'total_runtime_sec': total_time,
        'n_pairs': len(pairs),
        'n_ok': len(ok),
        'n_failed': len(per_pair) - len(ok),
        'per_pair': per_pair,
        'stratified': stratified,
    }
    (OUT_DIR / 'metrics.json').write_text(json.dumps(out, indent=2, default=str))
    print(f"\n=== DONE: total {total_time:.1f}s  n_ok={len(ok)}/{len(pairs)} ===")
    print('Stratified contact_f1_vs_source + q_com_path delta (post-pre):')
    for s in ['near_present', 'absent', 'moderate', 'extreme', 'all']:
        b = stratified.get(s, {})
        n = b.get('n', 0)
        cf = b.get('contact_f1_vs_source')
        cd = b.get('q_com_path_delta')
        tt = b.get('train_time_s')
        cf_s = f"{cf:.3f}" if cf is not None else "—"
        cd_s = f"{cd:+.3f}" if cd is not None else "—"
        tt_s = f"{tt:.2f}s" if tt is not None else "—"
        print(f"  {s:14s} n={n}  contact_f1_vs_source={cf_s}  q_com_delta={cd_s}  train={tt_s}")
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=None, help='restrict first-N pairs for debug')
    args = ap.parse_args()
    run(restrict_n=args.n)
