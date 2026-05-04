"""Minimal-Correspondence Escape v2 (JSON-authored, DTW-aligned, +MLP adapter).

Reads authored correspondences from
  eval/quotient_assets/minimal_corr_pairings.json
(5 pairs: 13, 14, 15, 17, 27 — the current authoring set).

Pipeline per pair (per task spec):
  1. Load authored correspondences from JSON.
  2. Resolve anchor motion: real-filename lookup under motions/ directory.
     Document the choice in metrics['anchor_motion_chosen'] and
     metrics['anchor_motion_choice_reason'].
  3. DTW-align source -> anchor timing using paired-bone positions
     (cosine on stacked paired-bone position triples per frame).
  4. For each source frame, find the DTW-mapped anchor frame best matching
     the paired-bone configuration.
  5. Concatenate retrieved anchor frames in source temporal order; hard-
     project source paired-bone signal onto target slots (pos/rot/vel/contact)
     with scale ratio; smooth by 3-tap moving average over joint positions.
  6. Optional: train a tiny 2-layer MLP residual (hidden=32, 200 steps Adam)
     that predicts joint-position correction from blended motion (pair-
     specific). Save the corrected motion to a SEPARATE adapter directory.

Outputs:
  eval/results/k_compare/minimal_corr/pair_<id>_<src>_to_<tgt>.npy (+ metrics.json)
  eval/results/k_compare/minimal_corr_adapter/pair_<id>_<src>_to_<tgt>.npy (+ metrics.json)
"""
from __future__ import annotations

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(str(PROJECT_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MOTION_DIR = ROOT / 'dataset/truebones/zoo/truebones_processed/motions'
COND_PATH = ROOT / 'dataset/truebones/zoo/truebones_processed/cond.npy'
CONTACT_GROUPS_PATH = ROOT / 'eval/quotient_assets/contact_groups.json'
EVAL_PAIRS_PATH = ROOT / 'idea-stage/eval_pairs.json'
AUTHORING_JSON = ROOT / 'eval/quotient_assets/minimal_corr_pairings.json'
OUT_DIR_RAW = ROOT / 'eval/results/k_compare/minimal_corr'
OUT_DIR_ADAPTER = ROOT / 'eval/results/k_compare/minimal_corr_adapter'
OUT_DIR_RAW.mkdir(parents=True, exist_ok=True)
OUT_DIR_ADAPTER.mkdir(parents=True, exist_ok=True)
CLF_V2 = ROOT / 'save/external_classifier_v2.pt'

FOOT_CH_IDX = 12
POS_SLICE = slice(0, 3)
ROT_SLICE = slice(3, 9)
VEL_SLICE = slice(9, 12)
SMOOTH_WIN = 3  # 3-tap moving average, per spec


# ==== Anchor-motion resolution =============================================
# JSON placeholder: "<TargetSkel>___<Action>_<closest>.npy". We scan motions/,
# filter on target_skel prefix, prefer keyword from rationale, pick lowest id
# of matching action; if none, fall back to any ground/idle clip.
#
# Per hard rule: do NOT silently substitute — record (candidates, chosen, reason).
ACTION_FALLBACK_KEYWORDS = {
    # Extraction of preferred action keywords per pair from the JSON rationale.
    # None => use the literal action from 'anchor_motion' field.
    'pair_13_Crab_eat_to_PolarBear':      ['Idle'],
    'pair_14_Comodoa_walk_to_Eagle':      ['Landing', 'TakeOff', 'IdleNest'],
    'pair_15_Goat_eat_to_Ant':            ['Idle'],
    'pair_17_Parrot_fly_to_PolarBearB':   ['Walk', 'Idle'],
    'pair_27_Buzzard_fly_to_Centipede':   ['Run'],
}


def _motion_id(fname: str) -> int:
    """Extract trailing integer id from 'Skel___Action_<id>.npy'."""
    m = re.search(r'_(\d+)\.npy$', fname)
    return int(m.group(1)) if m else 10**9


def _motion_frames(fname: str) -> int:
    try:
        return int(np.load(MOTION_DIR / fname, mmap_mode='r').shape[0])
    except Exception:
        return 0


def pick_anchor_motion(target_skel: str, preferred_keywords):
    """Pick actual filename from motions/ obeying the task's tiebreakers.

    Tiebreakers: (1) filename starts with '<target_skel>___'; (2) contains one
    of preferred_keywords; (3) lowest numeric id; (4) longest clip among ties.
    Returns (chosen_fname, all_candidates, reason_string).
    """
    all_files = sorted(os.listdir(MOTION_DIR))
    # Strict target-prefix filter ('Skel___' with triple underscore, per
    # Truebones naming). Both 'PolarBear' and 'PolarBearB' MUST be separated.
    prefix = f'{target_skel}___'
    cands = [f for f in all_files if f.startswith(prefix) and f.endswith('.npy')]
    # Also guard PolarBear vs PolarBearB: 'PolarBear___...' must NOT include PolarBearB
    if target_skel == 'PolarBear':
        cands = [f for f in cands if not f.startswith('PolarBearB___')]

    def keyword_rank(fname):
        for i, kw in enumerate(preferred_keywords):
            if kw.lower() in fname.lower():
                return i
        return len(preferred_keywords)

    cands_ranked = sorted(
        cands,
        key=lambda f: (keyword_rank(f), _motion_id(f), -_motion_frames(f))
    )
    if not cands_ranked:
        raise RuntimeError(f'No motions for target skel {target_skel}')
    chosen = cands_ranked[0]
    reason = (
        f'Filtered to {target_skel}___ prefix ({len(cands)} clips); '
        f'preferred keywords {preferred_keywords}; picked lowest-id match '
        f'({_motion_id(chosen)}), {_motion_frames(chosen)} frames.'
    )
    return chosen, cands, reason


# ==== Motion I/O ============================================================
def load_motion(fname, cond_entry):
    m = np.load(MOTION_DIR / fname)
    J = len(cond_entry['joints_names'])
    return m[:, :J, :].astype(np.float32)


def denormalize_motion(m_norm, cond_entry):
    J = m_norm.shape[1]
    std = cond_entry['std'][:J]
    mean = cond_entry['mean'][:J]
    if np.abs(m_norm).max() < 5:
        return m_norm * std + mean
    return m_norm


def body_scale(cond_entry):
    return float(np.linalg.norm(cond_entry['offsets'], axis=1).sum() + 1e-6)


# ==== DTW alignment using paired-bone positions (cosine) ====================
def _paired_pos_feature(motion_phys, idxs, body_scale_hint):
    T = motion_phys.shape[0]
    parts = []
    for j in idxs:
        if j >= motion_phys.shape[1]:
            parts.append(np.zeros((T, 3), dtype=np.float32))
        else:
            parts.append(motion_phys[:, j, POS_SLICE] / max(body_scale_hint, 1e-6))
    return np.concatenate(parts, axis=-1).astype(np.float32)  # [T, 3*K]


def dtw_align(src_feat, anc_feat):
    """Classical DTW with cosine cost; return per-src-frame anchor index.

    Returns (mapping[T_src], stretch_factor, path_len).
    mapping[t] = index into anc_feat aligned to src frame t.
    """
    Ts = src_feat.shape[0]; Ta = anc_feat.shape[0]
    # cosine distance = 1 - cos
    s = src_feat / (np.linalg.norm(src_feat, axis=-1, keepdims=True) + 1e-8)
    a = anc_feat / (np.linalg.norm(anc_feat, axis=-1, keepdims=True) + 1e-8)
    cost = 1.0 - s @ a.T  # [Ts, Ta]

    D = np.full((Ts + 1, Ta + 1), np.inf, dtype=np.float32)
    D[0, 0] = 0.0
    bt = np.zeros((Ts, Ta), dtype=np.int8)  # 0=diag,1=up,2=left
    for i in range(1, Ts + 1):
        for j in range(1, Ta + 1):
            c = cost[i - 1, j - 1]
            best = D[i - 1, j - 1]; k = 0
            if D[i - 1, j] < best: best = D[i - 1, j]; k = 1
            if D[i, j - 1] < best: best = D[i, j - 1]; k = 2
            D[i, j] = c + best
            bt[i - 1, j - 1] = k
    # Backtrack
    i, j = Ts - 1, Ta - 1
    path = []
    while i >= 0 and j >= 0:
        path.append((i, j))
        k = bt[i, j]
        if k == 0: i -= 1; j -= 1
        elif k == 1: i -= 1
        else: j -= 1
    path.reverse()
    # For each src frame pick median j across path elements
    mapping = np.zeros(Ts, dtype=np.int64)
    buckets = [[] for _ in range(Ts)]
    for ii, jj in path:
        buckets[ii].append(jj)
    for t in range(Ts):
        if buckets[t]:
            mapping[t] = int(np.median(buckets[t]))
        else:
            mapping[t] = mapping[t - 1] if t > 0 else 0
    stretch = Ta / max(Ts, 1)
    return mapping, float(stretch), len(path)


# ==== Retrieval + projection ================================================
def ma_smooth_positions(arr3, win=3):
    """3-tap moving average on joint POSITION channel (last-axis: 0:3)."""
    if win <= 1:
        return arr3
    T, J, D = arr3.shape
    pos = arr3[:, :, POS_SLICE].copy()  # [T, J, 3]
    k = win
    pad = k // 2
    padded = np.pad(pos, ((pad, pad), (0, 0), (0, 0)), mode='edge')
    sm = np.zeros_like(pos)
    for t in range(T):
        sm[t] = padded[t:t + k].mean(axis=0)
    out = arr3.copy()
    out[:, :, POS_SLICE] = sm
    return out


def retarget_dtw_project(src_phys, anchor_phys, src_idxs, tgt_idxs,
                          src_bs, tgt_bs):
    """DTW-align then hard-project paired-bone source signal onto target."""
    src_feat = _paired_pos_feature(src_phys, src_idxs, src_bs)
    anc_feat = _paired_pos_feature(anchor_phys, tgt_idxs, tgt_bs)
    mapping, stretch, plen = dtw_align(src_feat, anc_feat)

    T_s = src_phys.shape[0]
    retarget = anchor_phys[mapping].copy()  # [T_s, J_t, 13]

    scale_ratio = tgt_bs / max(src_bs, 1e-6)
    for si, ti in zip(src_idxs, tgt_idxs):
        if si >= src_phys.shape[1] or ti >= retarget.shape[1]:
            continue
        retarget[:, ti, POS_SLICE] = src_phys[:, si, POS_SLICE] * scale_ratio
        retarget[:, ti, ROT_SLICE] = src_phys[:, si, ROT_SLICE]
        retarget[:, ti, VEL_SLICE] = src_phys[:, si, VEL_SLICE] * scale_ratio
        retarget[:, ti, FOOT_CH_IDX] = src_phys[:, si, FOOT_CH_IDX]

    # Force root to source root if paired
    if src_idxs[0] == 0 and tgt_idxs[0] == 0:
        retarget[:, 0, POS_SLICE] = src_phys[:, 0, POS_SLICE] * scale_ratio
        retarget[:, 0, VEL_SLICE] = src_phys[:, 0, VEL_SLICE] * scale_ratio
        retarget[:, 0, ROT_SLICE] = src_phys[:, 0, ROT_SLICE]

    retarget = ma_smooth_positions(retarget, win=SMOOTH_WIN)
    return retarget.astype(np.float32), mapping, stretch, plen


# ==== Pair-specific tiny MLP adapter =======================================
class TinyPosAdapter(nn.Module):
    def __init__(self, J, hidden=32):
        super().__init__()
        self.J = J
        self.net = nn.Sequential(
            nn.Linear(J * 3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, J * 3),
        )

    def forward(self, pos):  # pos [T, J, 3]
        T = pos.shape[0]
        x = pos.reshape(T, -1)
        d = self.net(x).reshape(T, self.J, 3)
        return pos + d  # residual


def train_adapter_and_apply(retarget_phys, src_phys, src_idxs, tgt_idxs,
                             scale_ratio, steps=200, hidden=32, lr=1e-3):
    """Fit a pair-specific residual adapter that nudges paired-target-bone
    positions toward scaled source positions while regularizing unpaired joints
    to stay near the blended retrieval. This is the "adapter-corrected" output.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    T, J, _ = retarget_phys.shape
    pos_in = torch.tensor(retarget_phys[:, :, 0:3], dtype=torch.float32, device=device)
    tgt_full = pos_in.clone()
    for si, ti in zip(src_idxs, tgt_idxs):
        if si < src_phys.shape[1] and ti < J:
            tgt_full[:, ti] = torch.tensor(
                src_phys[:, si, 0:3] * scale_ratio, dtype=torch.float32, device=device
            )
    # Weight: paired joints get full weight; others get small regularization
    w = torch.ones(J, device=device) * 0.05
    for ti in tgt_idxs:
        if ti < J:
            w[ti] = 1.0

    adapter = TinyPosAdapter(J, hidden=hidden).to(device)
    opt = torch.optim.Adam(adapter.parameters(), lr=lr)
    for step in range(steps):
        pred = adapter(pos_in)
        err = (pred - tgt_full).pow(2).sum(-1)  # [T, J]
        loss = (err * w[None]).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = adapter(pos_in).cpu().numpy()
    out = retarget_phys.copy()
    out[:, :, 0:3] = pred
    # Re-apply smoothing after adapter to suppress residual jitter
    out = ma_smooth_positions(out, win=SMOOTH_WIN)
    return out.astype(np.float32), float(loss.item())


# ==== Eval helpers (classifier + Q + contact) ==============================
def classify_motion(clf, motion_phys, skel_cond, extract_fn, recover_fn):
    J_skel = skel_cond['offsets'].shape[0]
    m = motion_phys
    if m.shape[1] > J_skel:
        m = m[:, :J_skel]
    try:
        positions = recover_fn(m.astype(np.float32))
    except Exception:
        return None
    parents = skel_cond['parents'][:J_skel]
    feats = extract_fn(positions, parents)
    if feats is None or feats.shape[0] < 4:
        return None
    return clf.predict_label(feats)


def contact_f1(pred_sched, gt_sched, thresh=0.5):
    pred = (np.asarray(pred_sched) >= thresh).astype(np.int8).ravel()
    gt = (np.asarray(gt_sched) >= thresh).astype(np.int8).ravel()
    n = min(pred.size, gt.size)
    pred, gt = pred[:n], gt[:n]
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    p = tp / (tp + fp + 1e-8); r = tp / (tp + fn + 1e-8)
    return float(2 * p * r / (p + r + 1e-8))


def _l2(a, b):
    a = np.asarray(a); b = np.asarray(b)
    T_ = min(a.shape[0], b.shape[0])
    return float(np.linalg.norm((a[:T_] - b[:T_]).reshape(-1)))


def acc_smoothness(motion_phys):
    """Mean per-joint acceleration magnitude (smaller = smoother)."""
    pos = motion_phys[:, :, POS_SLICE]  # [T, J, 3]
    vel = np.diff(pos, axis=0, prepend=pos[:1])
    acc = np.diff(vel, axis=0, prepend=vel[:1])
    return float(np.linalg.norm(acc, axis=-1).mean())


def skating_metric(motion_phys):
    """Simple skating proxy: |velocity of joints that are in contact|.mean()."""
    vel = motion_phys[:, :, VEL_SLICE]  # [T, J, 3]
    con = motion_phys[:, :, FOOT_CH_IDX] > 0.5  # [T, J]
    v = np.linalg.norm(vel, axis=-1)
    if con.sum() == 0:
        return 0.0
    return float((v[con]).mean())


# ==== Main ==================================================================
def main():
    cond = np.load(COND_PATH, allow_pickle=True).item()
    with open(CONTACT_GROUPS_PATH) as f:
        contact_groups = json.load(f)
    with open(EVAL_PAIRS_PATH) as f:
        eval_pairs = json.load(f)['pairs']
    pair_by_id = {p['pair_id']: p for p in eval_pairs}
    with open(AUTHORING_JSON) as f:
        authoring = json.load(f)

    from eval.external_classifier import extract_classifier_features, ACTION_CLASSES  # noqa
    from eval.train_external_classifier_v2 import V2Classifier
    from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np
    from eval.quotient_extractor import extract_quotient

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clf = V2Classifier(str(CLF_V2), device=device)
    print(f'Loaded V2 classifier (arch={clf.arch}) on {device}')

    # Build pair list from authoring JSON (skip comment keys)
    ordered_keys = [k for k in authoring.keys() if k.startswith('pair_')]

    results_raw = {
        'config': {
            'method': 'minimal_correspondence_dtw_retrieval',
            'authoring_json': str(AUTHORING_JSON),
            'pair_keys': ordered_keys,
            'smooth_win_frames': SMOOTH_WIN,
            'classifier_ckpt': str(CLF_V2),
        },
        'pairs': [],
    }
    results_adapter = {
        'config': {
            'method': 'minimal_correspondence_dtw_retrieval_plus_adapter',
            'authoring_json': str(AUTHORING_JSON),
            'pair_keys': ordered_keys,
            'adapter': {'hidden': 32, 'steps': 200, 'lr': 1e-3},
            'smooth_win_frames': SMOOTH_WIN,
            'classifier_ckpt': str(CLF_V2),
        },
        'pairs': [],
    }

    t_all = time.time()
    for key in ordered_keys:
        entry = authoring[key]
        m = re.match(r'pair_(\d+)_', key)
        pid = int(m.group(1)) if m else -1
        src_skel = entry['source_skel']; tgt_skel = entry['target_skel']
        src_action = entry['source_action']
        pair_meta = pair_by_id.get(pid, {})
        src_fname = pair_meta.get('source_fname', '')
        family_gap = pair_meta.get('family_gap', 'unknown')

        pairs_list = entry['pairs']
        src_idxs = [p['src'] for p in pairs_list]
        tgt_idxs = [p['tgt'] for p in pairs_list]
        semantics = [p['semantic'] for p in pairs_list]

        print(f'\n=== {key}: {src_skel}({src_action}) -> {tgt_skel} [{family_gap}] ===')
        print(f'  authored {len(pairs_list)} correspondences')

        # Resolve anchor motion (real file on disk)
        pref_kws = ACTION_FALLBACK_KEYWORDS.get(key, [src_action])
        anchor_fname, anchor_cands, anchor_reason = pick_anchor_motion(tgt_skel, pref_kws)
        print(f'  anchor chosen: {anchor_fname}')
        print(f'  anchor reason: {anchor_reason}')

        rec_common = {
            'pair_id': pid, 'pair_key': key,
            'src_fname': src_fname, 'src_skel': src_skel,
            'src_action': src_action, 'tgt_skel': tgt_skel,
            'family_gap': family_gap,
            'n_paired_bones': len(pairs_list),
            'paired_semantics': semantics,
            'src_idxs': src_idxs, 'tgt_idxs': tgt_idxs,
            'anchor_motion_chosen': anchor_fname,
            'anchor_motion_choice_reason': anchor_reason,
            'anchor_motion_candidates': anchor_cands,
            'anchor_motion_authoring_placeholder': entry.get('anchor_motion', ''),
        }

        rec_raw = dict(rec_common); rec_adp = dict(rec_common)
        t_pair = time.time()
        try:
            src_norm = load_motion(src_fname, cond[src_skel])
            src_phys = denormalize_motion(src_norm, cond[src_skel])
            anc_norm = load_motion(anchor_fname, cond[tgt_skel])
            anc_phys = denormalize_motion(anc_norm, cond[tgt_skel])

            src_bs = body_scale(cond[src_skel]); tgt_bs = body_scale(cond[tgt_skel])
            for rec in (rec_raw, rec_adp):
                rec['src_body_scale'] = src_bs; rec['tgt_body_scale'] = tgt_bs
                rec['body_scale_ratio'] = tgt_bs / max(src_bs, 1e-6)
                rec['src_frames'] = int(src_phys.shape[0])
                rec['anchor_frames'] = int(anc_phys.shape[0])

            # DTW retarget
            retarget_phys, mapping, stretch, plen = retarget_dtw_project(
                src_phys, anc_phys, src_idxs, tgt_idxs, src_bs, tgt_bs,
            )
            dtw_summary = {
                'stretch_factor': stretch,
                'mean_mapping': float(np.mean(mapping)),
                'path_len': plen,
                'unique_anchor_frames_used': int(len(set(mapping.tolist()))),
                'n_src_frames': int(src_phys.shape[0]),
                'n_anc_frames': int(anc_phys.shape[0]),
            }
            for rec in (rec_raw, rec_adp):
                rec['dtw'] = dtw_summary

            out_raw = OUT_DIR_RAW / f'pair_{pid:02d}_{src_skel}_to_{tgt_skel}.npy'
            np.save(out_raw, retarget_phys.astype(np.float32))
            rec_raw['out_path'] = str(out_raw)

            # Adapter
            scale_ratio = tgt_bs / max(src_bs, 1e-6)
            retarget_adp, adp_loss = train_adapter_and_apply(
                retarget_phys, src_phys, src_idxs, tgt_idxs, scale_ratio,
                steps=200, hidden=32, lr=1e-3,
            )
            out_adp = OUT_DIR_ADAPTER / f'pair_{pid:02d}_{src_skel}_to_{tgt_skel}.npy'
            np.save(out_adp, retarget_adp.astype(np.float32))
            rec_adp['out_path'] = str(out_adp)
            rec_adp['adapter_final_loss'] = adp_loss

            # --- Evaluate both ---
            def eval_one(retarget_variant, rec, variant_name):
                tgt_pred = classify_motion(
                    clf, retarget_variant, cond[tgt_skel],
                    extract_classifier_features, recover_from_bvh_ric_np,
                )
                src_pred = classify_motion(
                    clf, src_phys, cond[src_skel],
                    extract_classifier_features, recover_from_bvh_ric_np,
                )
                rec['tgt_pred'] = tgt_pred
                rec['src_classifier_pred'] = src_pred
                rec['label_match'] = bool(tgt_pred == src_action)
                rec['behavior_preserved'] = bool(
                    src_pred is not None and tgt_pred == src_pred
                )
                # Q-components via tmp file
                tmp_fname = f'__mc2_{variant_name}_p{pid:02d}.npy'
                tmp_path = MOTION_DIR / tmp_fname
                try:
                    np.save(tmp_path, retarget_variant.astype(np.float32))
                    q_src = extract_quotient(
                        src_fname, cond[src_skel],
                        contact_groups=contact_groups,
                        motion_dir=str(MOTION_DIR),
                    )
                    q_out = extract_quotient(
                        tmp_fname, cond[tgt_skel],
                        contact_groups=contact_groups,
                        motion_dir=str(MOTION_DIR),
                    )
                finally:
                    if tmp_path.exists():
                        try: tmp_path.unlink()
                        except Exception: pass
                ss = np.asarray(q_src['contact_sched'])
                rs = np.asarray(q_out['contact_sched'])
                T = min(ss.shape[0], rs.shape[0])
                idx = np.clip(np.linspace(0, ss.shape[0] - 1, T).astype(int), 0, ss.shape[0] - 1)
                ss_t = ss[idx]
                ss_agg = ss_t.sum(axis=1) if ss_t.ndim == 2 else ss_t
                rs_agg = rs.sum(axis=1) if rs.ndim == 2 else rs
                rec['contact_f1_vs_source'] = contact_f1(rs_agg, ss_agg)
                rec['q_com_path_l2'] = _l2(q_src['com_path'], q_out['com_path'])
                rec['q_heading_vel_l2'] = _l2(q_src['heading_vel'], q_out['heading_vel'])
                rec['q_cadence_abs_diff'] = float(abs(float(q_src['cadence']) - float(q_out['cadence'])))
                T_ = min(ss_agg.size, rs_agg.size)
                rec['q_contact_sched_l2'] = float(np.linalg.norm(ss_agg[:T_] - rs_agg[:T_]))
                rec['accel_smoothness'] = acc_smoothness(retarget_variant)
                rec['skating'] = skating_metric(retarget_variant)
                rec['status'] = 'ok'

            eval_one(retarget_phys, rec_raw, 'raw')
            eval_one(retarget_adp, rec_adp, 'adapter')

            for rec in (rec_raw, rec_adp):
                rec['wall_time_s'] = float(time.time() - t_pair)

            print(f'  RAW: tgt_pred={rec_raw["tgt_pred"]} '
                  f'lm={rec_raw["label_match"]} cF1={rec_raw["contact_f1_vs_source"]:.3f}')
            print(f'  ADP: tgt_pred={rec_adp["tgt_pred"]} '
                  f'lm={rec_adp["label_match"]} cF1={rec_adp["contact_f1_vs_source"]:.3f}')
        except Exception as e:
            traceback.print_exc()
            rec_raw['status'] = 'error'; rec_raw['error'] = str(e)
            rec_adp['status'] = 'error'; rec_adp['error'] = str(e)

        results_raw['pairs'].append(rec_raw)
        results_adapter['pairs'].append(rec_adp)

    def summarize(res, label):
        ok = [p for p in res['pairs'] if p.get('status') == 'ok']
        preds = [p['tgt_pred'] for p in ok if p.get('tgt_pred') is not None]
        pred_counts = {k: preds.count(k) for k in sorted(set(preds))}
        s = {
            'n_authored': len(res['pairs']),
            'n_ok': len(ok),
            'mean_label_match': float(np.mean([int(p['label_match']) for p in ok])) if ok else 0.0,
            'mean_behavior_preserved': float(np.mean([int(p['behavior_preserved']) for p in ok])) if ok else 0.0,
            'mean_contact_f1_vs_source': float(np.mean([p['contact_f1_vs_source'] for p in ok])) if ok else None,
            'median_q_com_path_l2': float(np.median([p['q_com_path_l2'] for p in ok])) if ok else None,
            'median_q_heading_vel_l2': float(np.median([p['q_heading_vel_l2'] for p in ok])) if ok else None,
            'median_q_contact_sched_l2': float(np.median([p['q_contact_sched_l2'] for p in ok])) if ok else None,
            'median_q_cadence_abs_diff': float(np.median([p['q_cadence_abs_diff'] for p in ok])) if ok else None,
            'mean_accel_smoothness': float(np.mean([p['accel_smoothness'] for p in ok])) if ok else None,
            'mean_skating': float(np.mean([p['skating'] for p in ok])) if ok else None,
            'predicted_classes_distribution': pred_counts,
            'n_distinct_predicted_classes': len(pred_counts),
            'codex_verdict': {
                'threshold_label_match_0.25': (float(np.mean([int(p['label_match']) for p in ok])) if ok else 0) >= 0.25,
                'threshold_contact_f1_0.35': (float(np.mean([p['contact_f1_vs_source'] for p in ok])) if ok else 0) >= 0.35,
                'threshold_diversity_3classes': len(pred_counts) >= 3,
            },
            'baseline_context': {
                'prior_absent_label_match_ceiling_v2clf': 0.083,
                'prior_absent_contact_f1_K': 0.067,
                'prior_absent_contact_f1_H_v4': 0.459,
                'prior_absent_contact_f1_M2M_lite': 0.362,
            },
        }
        s['codex_verdict']['all_passed'] = all([
            s['codex_verdict']['threshold_label_match_0.25'],
            s['codex_verdict']['threshold_contact_f1_0.35'],
            s['codex_verdict']['threshold_diversity_3classes'],
        ])
        res['summary'] = s
        print(f'\n======== SUMMARY ({label}) ========')
        print(json.dumps(s, indent=2, default=float))

    summarize(results_raw, 'RAW')
    summarize(results_adapter, 'ADAPTER')
    results_raw['runtime_s'] = time.time() - t_all
    results_adapter['runtime_s'] = time.time() - t_all

    with open(OUT_DIR_RAW / 'metrics.json', 'w') as f:
        json.dump(results_raw, f, indent=2, default=float)
    with open(OUT_DIR_ADAPTER / 'metrics.json', 'w') as f:
        json.dump(results_adapter, f, indent=2, default=float)

    # Update unified_method_comparison_v2_full.json
    unified_path = ROOT / 'idea-stage/unified_method_comparison_v2_full.json'
    try:
        with open(unified_path) as f:
            unified = json.load(f)
    except Exception:
        unified = {}
    def row(res):
        s = res['summary']
        return {
            'absent_label_match': s['mean_label_match'],
            'absent_behavior_preserved': s['mean_behavior_preserved'],
            'absent_contact_f1_vs_source': s['mean_contact_f1_vs_source'],
            'absent_predicted_classes': s['predicted_classes_distribution'],
            'n_distinct_predicted_classes': s['n_distinct_predicted_classes'],
            'n_pairs_evaluated': s['n_ok'],
            'codex_verdict': s['codex_verdict'],
            'source': str(Path(res['config'].get('authoring_json', ''))),
        }
    unified['minimal_corr'] = row(results_raw)
    unified['minimal_corr_adapter'] = row(results_adapter)
    with open(unified_path, 'w') as f:
        json.dump(unified, f, indent=2, default=float)
    print(f'\nWrote updates to {unified_path}')


if __name__ == '__main__':
    main()
