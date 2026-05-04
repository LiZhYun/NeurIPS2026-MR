"""Extract z_tgt_avg latents from a MoReFlow checkpoint, for gauge analysis.

Mirrors sample/moreflow_retarget.py except saves the per-query continuous z
(BEFORE codebook quantization) instead of decoded motion. The latent of interest
is the z_tgt_avg [T_token_full, codebook_dim] output by Heun integration.

For fair cross-seed gauge comparison, fixes z_init RNG via --seed_for_z_init.
(MoReFlow's Heun starts from N(0, I) noise; same noise across seeds isolates the
trained generator's gauge orbit from sampling variance.)

Usage:
    python /tmp/save_moreflow_latents.py \
        --ckpt save/moreflow_flow/primary_70/ckpt_step000200000.pt \
        --manifest eval/benchmark_v3/queries_sif_intersection/manifest.json \
        --out_dir /tmp/moreflow_latents/primary_70
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loaders.truebones.truebones_utils.param_utils import (
    OBJECT_SUBSETS_DICT,
)
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
from sample.moreflow_retarget import (
    heun_integrate_ensemble, MOTION_DIR,
    WINDOW, STRIDE, TOKENS_PER_WINDOW,
    DEFAULT_INFERENCE_CONDS, DEFAULT_CFG_GAMMA, DEFAULT_HEUN_STEPS,
)


def extract_latent_one_query(query, model, skel_enc, registry, skel_features, max_J,
                              skel_to_id, skel_descs, args, device, z_init_rng):
    """Returns z_tgt_avg [T_token_full, codebook_dim] (numpy)."""
    src_skel = query['skel_a']
    tgt_skel = query['skel_b']
    src_fname = query['src_fname']

    UNK_ID = len(skel_to_id)
    src_id = skel_to_id.get(src_skel, UNK_ID)
    tgt_id = skel_to_id.get(tgt_skel, UNK_ID)
    if src_id == UNK_ID: src_id = 0
    if tgt_id == UNK_ID: tgt_id = 0

    src_pj_pad, src_pj_mask = pad_to_max_joints(skel_features[src_skel]['per_joint'], max_J)
    tgt_pj_pad, tgt_pj_mask = pad_to_max_joints(skel_features[tgt_skel]['per_joint'], max_J)
    src_pj = src_pj_pad.unsqueeze(0).to(device)
    src_m = src_pj_mask.unsqueeze(0).to(device)
    src_a = skel_features[src_skel]['agg'].unsqueeze(0).to(device)
    tgt_pj = tgt_pj_pad.unsqueeze(0).to(device)
    tgt_m = tgt_pj_mask.unsqueeze(0).to(device)
    tgt_a = skel_features[tgt_skel]['agg'].unsqueeze(0).to(device)
    src_graph = skel_enc(src_pj, src_m, src_a)
    tgt_graph = skel_enc(tgt_pj, tgt_m, tgt_a)

    motion_phys = np.load(MOTION_DIR / src_fname).astype(np.float32)
    T = motion_phys.shape[0]
    T_crop = (T // STRIDE) * STRIDE
    motion_phys = motion_phys[:T_crop]
    motion_t = torch.from_numpy(motion_phys).to(device)

    motion_norm = registry.normalize(src_skel, motion_t.unsqueeze(0))
    z_src_full, _ = registry.encode_window(src_skel, motion_norm.squeeze(0))
    z_src_full = z_src_full.squeeze(0)
    T_token_full = z_src_full.shape[0]

    chunk_token_stride = TOKENS_PER_WINDOW // 2
    if T_token_full <= TOKENS_PER_WINDOW:
        chunk_starts = [0]
    else:
        chunk_starts = list(range(0, T_token_full - TOKENS_PER_WINDOW + 1, chunk_token_stride))
        last_start = T_token_full - TOKENS_PER_WINDOW
        if chunk_starts[-1] != last_start:
            chunk_starts.append(last_start)

    codebook_dim = registry.get(tgt_skel)['model'].codebook_dim
    z_tgt_acc = torch.zeros(T_token_full, codebook_dim, device=device)
    z_tgt_count = torch.zeros(T_token_full, device=device)

    # heun_integrate_ensemble samples z_init internally — to fix RNG we monkey-patch
    # torch.randn for the duration of this query, using a CPU generator seeded from z_init_rng.
    # Simpler: pre-generate z_init per chunk and inject it via a new sample function.
    # heun_integrate_ensemble accepts z_init as third positional arg (per sig at line 57:
    #   heun_integrate_ensemble(model, skel_enc, z_init, src_id, tgt_id, src_graph, tgt_graph, ...))
    # That means we PASS z_init in. For fair gauge comparison, draw one per chunk from our seeded RNG.

    for chunk_start in chunk_starts:
        chunk_end = min(chunk_start + TOKENS_PER_WINDOW, T_token_full)
        chunk_z = z_src_full[chunk_start:chunk_end]
        if chunk_end - chunk_start < TOKENS_PER_WINDOW:
            pad = TOKENS_PER_WINDOW - chunk_z.shape[0]
            chunk_z = torch.cat([chunk_z, torch.zeros(pad, chunk_z.shape[1], device=device)])
        chunk_z = chunk_z.unsqueeze(0)

        # Seeded init noise (same shape as chunk_z's target latent: [1, 8, codebook_dim])
        z_init_np = z_init_rng.randn(1, TOKENS_PER_WINDOW, codebook_dim).astype(np.float32)
        z_init = torch.from_numpy(z_init_np).to(device)

        chunk_frame_start = chunk_start * STRIDE
        chunk_frame_end = chunk_frame_start + WINDOW
        if chunk_frame_end > T_crop:
            chunk_frame_end = T_crop
            chunk_frame_start = T_crop - WINDOW
        chunk_motion_phys = motion_phys[chunk_frame_start:chunk_frame_end]

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
            model, skel_enc, z_init, src_id, tgt_id, src_graph, tgt_graph,
            cond_type_ints, cond_vecs,
            args.heun_steps, args.cfg_gamma, device,
        ).squeeze(0)

        chunk_len_actual = chunk_end - chunk_start
        z_tgt_acc[chunk_start:chunk_end] += z_tgt_chunk[:chunk_len_actual]
        z_tgt_count[chunk_start:chunk_end] += 1.0

    z_tgt_avg = z_tgt_acc / z_tgt_count.clamp(min=1.0).unsqueeze(-1)
    return z_tgt_avg.cpu().numpy(), tgt_skel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--manifest', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--inference_conds', type=str, nargs='+', default=DEFAULT_INFERENCE_CONDS)
    parser.add_argument('--cfg_gamma', type=float, default=DEFAULT_CFG_GAMMA)
    parser.add_argument('--heun_steps', type=int, default=DEFAULT_HEUN_STEPS)
    parser.add_argument('--seed_for_z_init', type=int, default=42)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[mfl-latents] Device: {device}")
    print(f"[mfl-latents] Loading ckpt: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    skels = ckpt['skels']
    skel_to_id = ckpt['skel_to_id']
    print(f"[mfl-latents] Trained on {len(skels)} skels")

    model = DiscreteFlowTransformer(n_skels=len(skels)).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    skel_enc = SkelGraphEncoder().to(device)
    skel_enc.load_state_dict(ckpt['skel_enc'])
    skel_enc.eval()

    print("[mfl-latents] Loading Stage A registry for ALL 70 skels...")
    all_skels = list(set(OBJECT_SUBSETS_DICT['train_v3']) | set(OBJECT_SUBSETS_DICT['test_v3']))
    registry = StageARegistry(all_skels, device=device)
    from data_loaders.truebones.truebones_utils.param_utils import DATASET_DIR as _DD
    cond_dict = np.load(Path(_DD) / 'cond.npy', allow_pickle=True).item()
    skel_features = build_skel_features(cond_dict)
    max_J = max(skel_features[s]['n_joints'] for s in skels)
    contact_groups = load_contact_groups()
    skel_descs = precompute_skel_descriptors(cond_dict, contact_groups)

    print(f"[mfl-latents] Loading manifest: {args.manifest}")
    manifest = json.loads(Path(args.manifest).read_text())
    queries = manifest.get('queries', manifest.get('triples', []))
    if args.limit:
        queries = queries[:args.limit]
    print(f"[mfl-latents] Processing {len(queries)} queries (Heun {args.heun_steps} steps, "
          f"CFG γ={args.cfg_gamma}, conds={args.inference_conds})...")

    z_init_rng = np.random.RandomState(args.seed_for_z_init)
    triple_meta = []
    t0 = time.time()
    with torch.no_grad():
        for i, q in enumerate(queries):
            try:
                z_lat, tgt_skel = extract_latent_one_query(
                    q, model, skel_enc, registry, skel_features, max_J,
                    skel_to_id, skel_descs, args, device, z_init_rng)
                np.save(out_dir / f'z_query_{i:04d}.npy', z_lat)
                triple_meta.append({
                    'query_id': i, 'src_skel': q.get('skel_a'), 'tgt_skel': tgt_skel,
                    'src_fname': q.get('src_fname'),
                    'T_tokens': z_lat.shape[0], 'codebook_dim': z_lat.shape[1],
                })
            except Exception as e:
                print(f"  [{i+1}/{len(queries)}] FAILED: {type(e).__name__}: {e}")
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(queries)}] done ({time.time() - t0:.0f}s)")

    (out_dir / 'meta.json').write_text(json.dumps(triple_meta, indent=2))
    print(f"[mfl-latents] saved {len(triple_meta)} queries to {out_dir} in {time.time() - t0:.0f}s")


if __name__ == '__main__':
    main()
