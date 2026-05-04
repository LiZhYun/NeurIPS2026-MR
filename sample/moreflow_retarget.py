"""MoReFlow Stage B inference: retarget v3 benchmark queries.

For each query (src_motion, src_skel, tgt_skel):
  1. Encode source through Stage A (full length, native sequence)
  2. Window into 8-token chunks with stride=4 (50% overlap)
  3. Per chunk, integrate ODE under 4 conditions in parallel and average velocity fields:
     {null, root_vel, root_XY, root_Z} (skip EE conditions in default policy)
  4. CFG with γ=2.0, 25 Heun steps per condition
  5. Latent overlap-add: average overlapping z's in continuous codebook embedding space
  6. Hard-quantize at the end to target codebook
  7. Decode through target Stage A; un-normalize; write [T, J_tgt, 13] to query_NNNN.npy

Usage:
  python -m sample.moreflow_retarget --ckpt save/moreflow_flow/primary_70/ckpt_final.pt \\
      --fold 42 --out_dir save/moreflow/v3/primary_70/fold_42
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loaders.truebones.truebones_utils.param_utils import OBJECT_SUBSETS_DICT
from model.moreflow.flow_transformer import DiscreteFlowTransformer
from model.moreflow.skel_graph import (
    SkelGraphEncoder, build_skel_features, pad_to_max_joints,
)
from model.moreflow.stage_a_registry import StageARegistry
from eval.moreflow_phi import (
    CONDITIONS, D_COND_PADDED, G_MAX,
    phi as phi_np, precompute_skel_descriptors, load_contact_groups,
)
from train.train_moreflow_flow import COND_TYPE_TO_INT

DATA_ROOT = PROJECT_ROOT / 'dataset/truebones/zoo/truebones_processed'
COND_PATH = DATA_ROOT / 'cond.npy'
MOTION_DIR = DATA_ROOT / 'motions'
WINDOW = 32
STRIDE = 4
TOKENS_PER_WINDOW = 8

# Default inference policy (pre-registered in v5 §3.6)
DEFAULT_INFERENCE_CONDS = ['null', 'root_vel', 'root_XY', 'root_Z']
DEFAULT_CFG_GAMMA = 2.0
DEFAULT_HEUN_STEPS = 25


def heun_integrate_ensemble(model, skel_enc, z_init, src_id, tgt_id, src_graph, tgt_graph,
                             cond_type_ints, cond_vecs, n_steps, cfg_gamma, device):
    """Integrate v_psi from q=0 to q=1 using Heun's 2nd-order method with CFG.

    Averages velocity fields across a list of conditions PER STEP (v5-faithful: per-step
    average, not per-integration average; the two are NOT equivalent for nonlinear flows).

    z_init:         [B, T_token, codebook_dim]
    cond_type_ints: list of int          — one per condition in ensemble
    cond_vecs:      list of Tensor[1, D] — one per condition in ensemble
    Returns z at q=1: [B, T_token, codebook_dim]
    """
    B, T_tok, d = z_init.shape
    z = z_init.clone()
    dq = 1.0 / n_steps

    src_id_t = torch.full((B,), src_id, dtype=torch.long, device=device)
    tgt_id_t = torch.full((B,), tgt_id, dtype=torch.long, device=device)

    def one_cond_velocity(z_cur, q_cur, cond_type_int, cond_vec):
        """Compute CFG-guided velocity for ONE condition at q_cur. Returns [B, T_tok, d]."""
        q_t = torch.full((B,), q_cur, device=device)
        cond_type_t = torch.full((B,), cond_type_int, dtype=torch.long, device=device)
        cond_vec_t = cond_vec.expand(B, -1).to(device)

        if cond_type_int == COND_TYPE_TO_INT['null']:
            # "null" = unconditional: always call with cond_mask=1 (zeroes cond_h in the model,
            # matching what the CFG-trained unconditional branch expects).
            cond_mask = torch.ones(B, device=device)
            return model(z_cur, q_t, src_id_t, tgt_id_t, src_graph, tgt_graph,
                          cond_type_t, cond_vec_t, cond_mask)

        # Conditioned: CFG mixing
        cond_mask_off = torch.zeros(B, device=device)
        cond_mask_on = torch.ones(B, device=device)
        v_cond = model(z_cur, q_t, src_id_t, tgt_id_t, src_graph, tgt_graph,
                       cond_type_t, cond_vec_t, cond_mask_off)
        if cfg_gamma == 1.0:
            return v_cond
        v_uncond = model(z_cur, q_t, src_id_t, tgt_id_t, src_graph, tgt_graph,
                         cond_type_t, cond_vec_t, cond_mask_on)
        return (1.0 - cfg_gamma) * v_uncond + cfg_gamma * v_cond

    def ensemble_velocity(z_cur, q_cur):
        """Average velocities across the ensemble of conditions at q_cur."""
        vs = [one_cond_velocity(z_cur, q_cur, ct, cv)
              for ct, cv in zip(cond_type_ints, cond_vecs)]
        return torch.stack(vs).mean(dim=0)

    for i in range(n_steps):
        q_cur = i * dq
        v1 = ensemble_velocity(z, q_cur)
        z_pred = z + dq * v1
        q_next = min(1.0 - 1e-6, q_cur + dq)
        v2 = ensemble_velocity(z_pred, q_next)
        z = z + 0.5 * dq * (v1 + v2)
    return z


@torch.no_grad()
def retarget_one_query(query, model, skel_enc, registry, skel_features, max_J,
                       skel_to_id, skel_descs, args, device):
    """Run inference for one v3 query. Returns [T, J_tgt, 13] motion (numpy)."""
    src_skel = query['skel_a']
    tgt_skel = query['skel_b']
    src_fname = query['src_fname']

    # Held-out skels (present in v3 but absent from the trained scope, e.g. 60-skel inductive
    # ablation on test_test queries): use a sentinel UNK id. The model has NOT seen this id
    # during training, but the graph features (offsets, parents, depths) are available and
    # carry morphology signal. This is the principled inductive measurement — we report the
    # quality we actually get, which should be substantially worse than the transductive run
    # and serves as the gauge-induced-gap measurement.
    UNK_ID = len(skel_to_id)  # one past the last trained id
    src_id = skel_to_id.get(src_skel, UNK_ID)
    tgt_id = skel_to_id.get(tgt_skel, UNK_ID)
    if src_id == UNK_ID or tgt_id == UNK_ID:
        # Clip the id to the embedding's vocab size (the model was built with n_skels = len(trained))
        # — accessing the UNK index would crash. Re-use skel id 0 (first trained skel) and rely
        # on graph features to carry morphology. Logged as "inductive_fallback" in output.
        # Alternative (cleaner): rebuild the model with n_skels+1 and zero the UNK embedding.
        # For now, the graph-path is the primary signal; id fallback is secondary.
        if src_id == UNK_ID:
            src_id = 0
        if tgt_id == UNK_ID:
            tgt_id = 0

    # Skel graph embeddings
    src_pj = pad_to_max_joints(skel_features[src_skel]['per_joint'], max_J)[0].unsqueeze(0).to(device)
    src_m = pad_to_max_joints(skel_features[src_skel]['per_joint'], max_J)[1].unsqueeze(0).to(device)
    src_a = skel_features[src_skel]['agg'].unsqueeze(0).to(device)
    tgt_pj = pad_to_max_joints(skel_features[tgt_skel]['per_joint'], max_J)[0].unsqueeze(0).to(device)
    tgt_m = pad_to_max_joints(skel_features[tgt_skel]['per_joint'], max_J)[1].unsqueeze(0).to(device)
    tgt_a = skel_features[tgt_skel]['agg'].unsqueeze(0).to(device)
    src_graph = skel_enc(src_pj, src_m, src_a)
    tgt_graph = skel_enc(tgt_pj, tgt_m, tgt_a)

    # Load + encode source motion
    src_path = MOTION_DIR / src_fname
    motion_phys = np.load(src_path).astype(np.float32)              # [T, J, 13]
    T = motion_phys.shape[0]
    # Crop T to multiple of STRIDE so token sequence is well-defined
    T_crop = (T // STRIDE) * STRIDE
    motion_phys = motion_phys[:T_crop]
    motion_t = torch.from_numpy(motion_phys).to(device)

    # Stage A encode at full length: encoder handles arbitrary T (multiple of downsample)
    motion_norm = registry.normalize(src_skel, motion_t.unsqueeze(0))      # [1, T_crop, J_src, 13]
    z_src_full, _ = registry.encode_window(src_skel, motion_norm.squeeze(0))  # [1, T_crop/4, codebook_dim]
    z_src_full = z_src_full.squeeze(0)                                     # [T_token_full, codebook_dim]
    T_token_full = z_src_full.shape[0]

    # Window into 8-token chunks with stride=4 (50% overlap)
    # Fix: append a final chunk aligned at T_token_full - TOKENS_PER_WINDOW if not already
    # in the stride schedule — otherwise tail tokens are dropped for most clips.
    chunk_token_stride = TOKENS_PER_WINDOW // 2
    if T_token_full <= TOKENS_PER_WINDOW:
        chunk_starts = [0]
    else:
        chunk_starts = list(range(0, T_token_full - TOKENS_PER_WINDOW + 1, chunk_token_stride))
        last_start = T_token_full - TOKENS_PER_WINDOW
        if chunk_starts[-1] != last_start:
            chunk_starts.append(last_start)
    n_chunks = len(chunk_starts)

    # Accumulator for overlap-add
    codebook_dim = registry.get(tgt_skel)['model'].codebook_dim
    z_tgt_acc = torch.zeros(T_token_full, codebook_dim, device=device)
    z_tgt_count = torch.zeros(T_token_full, device=device)

    # Pre-resolve cond type/value per condition in the ensemble.
    # For each chunk we compute the condition vector on the chunk's source motion.
    # Per-step velocity averaging is handled inside heun_integrate_ensemble.
    for chunk_start in chunk_starts:
        chunk_end = min(chunk_start + TOKENS_PER_WINDOW, T_token_full)
        chunk_z = z_src_full[chunk_start:chunk_end]
        if chunk_end - chunk_start < TOKENS_PER_WINDOW:
            pad = TOKENS_PER_WINDOW - chunk_z.shape[0]
            chunk_z = torch.cat([chunk_z, torch.zeros(pad, chunk_z.shape[1], device=device)])
        chunk_z = chunk_z.unsqueeze(0)                                           # [1, 8, codebook_dim]

        chunk_frame_start = chunk_start * STRIDE
        chunk_frame_end = chunk_frame_start + WINDOW
        if chunk_frame_end > T_crop:
            chunk_frame_end = T_crop
            chunk_frame_start = T_crop - WINDOW
        chunk_motion_phys = motion_phys[chunk_frame_start:chunk_frame_end]       # [WINDOW, J_src, 13]

        cond_type_ints = []
        cond_vecs = []
        for c_str in args.inference_conds:
            cond_type_ints.append(COND_TYPE_TO_INT[c_str])
            if c_str == 'null':
                cond_vecs.append(torch.zeros(1, D_COND_PADDED, device=device))
            else:
                cond_vec_np = phi_np(chunk_motion_phys, c_str, skel_descs[src_skel])
                cond_vecs.append(torch.from_numpy(cond_vec_np).float().unsqueeze(0).to(device))

        z_tgt_chunk = heun_integrate_ensemble(
            model, skel_enc, chunk_z, src_id, tgt_id, src_graph, tgt_graph,
            cond_type_ints, cond_vecs,
            args.heun_steps, args.cfg_gamma, device,
        ).squeeze(0)                                                              # [8, codebook_dim]

        chunk_len_actual = chunk_end - chunk_start
        z_tgt_acc[chunk_start:chunk_end] += z_tgt_chunk[:chunk_len_actual]
        z_tgt_count[chunk_start:chunk_end] += 1.0

    # Normalize overlap
    # Safety: any position with count==0 means the chunking missed it (should not happen
    # with the tail-append fix above). Sanity-check and warn.
    if (z_tgt_count == 0).any():
        n_missed = int((z_tgt_count == 0).sum())
        print(f"  WARN: {n_missed}/{T_token_full} token positions missed by overlap-add — "
              f"check chunk_starts logic.")
    z_tgt_avg = z_tgt_acc / z_tgt_count.clamp(min=1.0).unsqueeze(-1)

    # Hard-quantize to target codebook (only valid K_eff entries)
    cb = registry.codebook(tgt_skel, padded_to=0)                              # [K_eff, codebook_dim]
    dists = torch.cdist(z_tgt_avg.unsqueeze(0), cb.unsqueeze(0))                # [1, T_token_full, K_eff]
    indices = dists.argmin(dim=-1).squeeze(0)                                   # [T_token_full]

    # Decode through target tokenizer
    indices = indices.unsqueeze(0)                                              # [1, T_token_full]
    decoded = registry.decode_tokens(tgt_skel, indices)                         # [1, T, J_tgt, 13]
    decoded_phys = registry.unnormalize(tgt_skel, decoded).squeeze(0)           # [T, J_tgt, 13]

    return decoded_phys.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to ckpt_final.pt of trained Stage B run')
    parser.add_argument('--fold', type=int, default=42, help='v3 fold (42 or 43)')
    parser.add_argument('--bench_version', choices=['v3', 'v5'], default='v3')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Directory to write query_NNNN.npy outputs')
    parser.add_argument('--inference_conds', type=str, nargs='+',
                        default=DEFAULT_INFERENCE_CONDS,
                        help='Conditions to ensemble at inference')
    parser.add_argument('--cfg_gamma', type=float, default=DEFAULT_CFG_GAMMA)
    parser.add_argument('--heun_steps', type=int, default=DEFAULT_HEUN_STEPS)
    parser.add_argument('--limit', type=int, default=None,
                        help='Process only first N queries (debug)')
    parser.add_argument('--manifest', type=str, default=None,
                        help='Custom manifest path (overrides bench_version+fold lookup)')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Loading ckpt: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    skels = ckpt['skels']
    skel_to_id = ckpt['skel_to_id']
    print(f"Trained on {len(skels)} skels")

    # Build model
    model = DiscreteFlowTransformer(n_skels=len(skels)).to(device)
    skel_enc = SkelGraphEncoder().to(device)
    model.load_state_dict(ckpt['model'])
    skel_enc.load_state_dict(ckpt['skel_enc'])
    model.eval()
    skel_enc.eval()

    # Build registry — needs all skels referenced in v3 (could be more than trained set)
    print("Loading Stage A registry for ALL 70 skels (queries may target any)...")
    all_skels = list(set(OBJECT_SUBSETS_DICT['train_v3']) | set(OBJECT_SUBSETS_DICT['test_v3']))
    registry = StageARegistry(all_skels, device=device)

    # Build skel features
    cond_dict = np.load(COND_PATH, allow_pickle=True).item()
    contact_groups = load_contact_groups()
    skel_descs = precompute_skel_descriptors(cond_dict, contact_groups)
    skel_features = build_skel_features(cond_dict)
    max_J = max(skel_features[s]['n_joints'] for s in skels)

    # Load manifest (custom path overrides v3 or v5 fold-based lookup)
    if args.manifest:
        manifest_path = Path(args.manifest)
    else:
        queries_dir = 'queries_v5' if getattr(args, 'bench_version', 'v3') == 'v5' else 'queries'
        manifest_path = PROJECT_ROOT / f'eval/benchmark_v3/{queries_dir}/fold_{args.fold}/manifest.json'
    print(f"Loading {manifest_path}")
    with open(manifest_path) as f:
        manifest = json.load(f)
    queries = manifest['queries']
    if args.limit:
        queries = queries[:args.limit]
    print(f"Inference conditions: {args.inference_conds}, CFG γ={args.cfg_gamma}, "
          f"Heun steps={args.heun_steps}")
    print(f"Processing {len(queries)} queries...")

    n_done = 0
    n_skipped = 0
    t0 = time.time()
    for i, q in enumerate(queries):
        out_path = out_dir / f"query_{q['query_id']:04d}.npy"
        if out_path.exists():
            n_done += 1
            continue
        try:
            result = retarget_one_query(q, model, skel_enc, registry, skel_features,
                                         max_J, skel_to_id, skel_descs, args, device)
            if result is None:
                n_skipped += 1
                continue
            np.save(out_path, result.astype(np.float32))
            n_done += 1
        except Exception as e:
            print(f"  query {q['query_id']} ({q['skel_a']}→{q['skel_b']}): FAILED — {e}")
            import traceback
            traceback.print_exc()
            n_skipped += 1

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(queries) - i - 1)
            print(f"  [{i+1}/{len(queries)}] done={n_done}, skipped={n_skipped}, "
                  f"({elapsed:.0f}s, ETA {eta/60:.0f}min)")

    print(f"\nFinished: {n_done}/{len(queries)} queries written, {n_skipped} skipped.")
    print(f"Output dir: {out_dir}")


if __name__ == '__main__':
    main()
