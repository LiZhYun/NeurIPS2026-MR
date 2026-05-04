"""Extract z_b latents from AL-Flow-Src checkpoint (source-conditioned variant).

Mirrors run_anchor_label_flow_src_v5.py but saves z_b instead of decoded motion.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loaders.truebones.truebones_utils.param_utils import OBJECT_SUBSETS_DICT, DATASET_DIR
from eval.benchmark_v3.action_taxonomy import ACTION_CLUSTERS, parse_action_from_filename, action_to_cluster
from model.actionbridge.generator_anchor_src import AnchorLabelFlowSrcGenerator
from model.moreflow.skel_graph import SkelGraphEncoder, build_skel_features, pad_to_max_joints
from model.moreflow.stage_a_registry import StageARegistry

CLUSTERS = sorted(ACTION_CLUSTERS.keys())
CLUSTER_TO_IDX = {c: i for i, c in enumerate(CLUSTERS)}
EXACT_ACTIONS_FILE = PROJECT_ROOT / 'eval/benchmark_v3/exact_action_idx.json'
EXACT_ACTION_TO_IDX = json.loads(EXACT_ACTIONS_FILE.read_text()) if EXACT_ACTIONS_FILE.exists() else {}

CACHE_WINDOW = 32
MOTION_DIR = Path(DATASET_DIR) / 'motions'


def encode_source_via_stage_a(stage_a, skel_a, motion_phys):
    T = motion_phys.shape[0]
    if T >= CACHE_WINDOW:
        motion_phys = motion_phys[:CACHE_WINDOW]
    else:
        pad = np.zeros((CACHE_WINDOW - T, *motion_phys.shape[1:]), dtype=np.float32)
        motion_phys = np.concatenate([motion_phys, pad], axis=0)
    motion_t = torch.from_numpy(motion_phys).unsqueeze(0).to(stage_a.device)
    motion_norm = stage_a.normalize(skel_a, motion_t)
    motion_norm = torch.nan_to_num(motion_norm, nan=0.0, posinf=0.0, neginf=0.0)
    z_src, _ = stage_a.encode_window(skel_a, motion_norm.squeeze(0))
    return z_src  # [1, T_token=8, codebook_dim=256]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--cfg_scale', type=float, default=1.0)
    p.add_argument('--seed_for_z_init', type=int, default=42)
    p.add_argument('--limit', type=int, default=None)
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[al-src] Loading ckpt: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    skels = ckpt['skels']; skel_to_id = ckpt['skel_to_id']

    G = AnchorLabelFlowSrcGenerator(
        n_skels=len(skels), n_clusters=len(CLUSTERS) + 1,
        n_exact_actions=ckpt.get('n_exact_actions', (max(EXACT_ACTION_TO_IDX.values())+1 if EXACT_ACTION_TO_IDX else 124) + 1),
    ).to(device)
    G.load_state_dict(ckpt['G']); G.eval()
    skel_enc = SkelGraphEncoder().to(device)
    skel_enc.load_state_dict(ckpt['skel_enc']); skel_enc.eval()

    print("[al-src] Loading registry...")
    all_skels = list(set(OBJECT_SUBSETS_DICT['train_v3']) | set(OBJECT_SUBSETS_DICT['test_v3']))
    registry = StageARegistry(all_skels, device=device)
    cond_dict = np.load(Path(DATASET_DIR) / 'cond.npy', allow_pickle=True).item()
    skel_features = build_skel_features(cond_dict)
    max_J = max(skel_features[s]['n_joints'] for s in skels)

    manifest = json.loads(Path(args.manifest).read_text())
    queries = manifest.get('queries', manifest.get('triples', []))
    if args.limit: queries = queries[:args.limit]
    print(f"[al-src] Processing {len(queries)} queries...")

    rng = np.random.RandomState(args.seed_for_z_init)
    triple_meta = []; t0 = time.time()
    with torch.no_grad():
        for i, q in enumerate(queries):
            try:
                src_skel = q['skel_a']; tgt_skel = q['skel_b']
                src_fname = q.get('src_fname', '')
                cluster_str = action_to_cluster(parse_action_from_filename(src_fname)) or ''
                cluster_id_int = CLUSTER_TO_IDX.get(cluster_str, 0) + 1  # +1 because 0 = NULL
                exact_id_int = EXACT_ACTION_TO_IDX.get(q.get('src_action', ''), 0)

                # Source latent via Stage A
                motion_phys = np.load(MOTION_DIR / src_fname).astype(np.float32)
                src_z = encode_source_via_stage_a(registry, src_skel, motion_phys)  # [1, 8, 256]

                src_id = torch.full((1,), skel_to_id.get(src_skel, 0), dtype=torch.long, device=device)
                tgt_id = torch.full((1,), skel_to_id.get(tgt_skel, 0), dtype=torch.long, device=device)
                src_pj_pad, src_pj_mask = pad_to_max_joints(skel_features[src_skel]['per_joint'], max_J)
                tgt_pj_pad, tgt_pj_mask = pad_to_max_joints(skel_features[tgt_skel]['per_joint'], max_J)
                src_pj_pad = src_pj_pad.unsqueeze(0).to(device)
                src_pj_mask = src_pj_mask.unsqueeze(0).to(device)
                src_a = skel_features[src_skel]['agg'].unsqueeze(0).to(device)
                tgt_pj_pad = tgt_pj_pad.unsqueeze(0).to(device)
                tgt_pj_mask = tgt_pj_mask.unsqueeze(0).to(device)
                tgt_a = skel_features[tgt_skel]['agg'].unsqueeze(0).to(device)
                src_graph = skel_enc(src_pj_pad, src_pj_mask, src_a)
                tgt_graph = skel_enc(tgt_pj_pad, tgt_pj_mask, tgt_a)
                cid = torch.full((1,), cluster_id_int, dtype=torch.long, device=device)
                eid = torch.full((1,), exact_id_int, dtype=torch.long, device=device)

                n_token = 8; codebook_dim = registry.get(tgt_skel)['model'].codebook_dim
                z_init_np = rng.randn(1, n_token, codebook_dim).astype(np.float32)
                z_init = torch.from_numpy(z_init_np).to(device)

                # Single-step Euler with optional CFG
                t = torch.zeros(1, device=device)
                v_cond = G(z_init, t, tgt_id, tgt_graph, src_z, src_id, src_graph, cid, eid)
                z_b = z_init + v_cond  # cfg_scale=1.0 simplification

                z_lat = z_b.squeeze(0).cpu().numpy()
                np.save(out_dir / f'z_query_{i:04d}.npy', z_lat)
                triple_meta.append({
                    'query_id': i, 'src_skel': src_skel, 'tgt_skel': tgt_skel,
                    'src_fname': src_fname, 'T_tokens': z_lat.shape[0],
                    'codebook_dim': z_lat.shape[1],
                    'cluster_id': cluster_id_int, 'exact_id': exact_id_int,
                })
            except Exception as e:
                print(f"  [{i+1}] FAILED: {type(e).__name__}: {e}")
            if (i+1) % 50 == 0:
                print(f"  [{i+1}/{len(queries)}] done ({time.time()-t0:.0f}s)")

    (out_dir / 'meta.json').write_text(json.dumps(triple_meta, indent=2))
    print(f"[al-src] saved {len(triple_meta)} queries to {out_dir} in {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
