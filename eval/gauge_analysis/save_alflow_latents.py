"""Extract z_b (AL-Flow generator output before Stage A decode) from a trained AL-Flow checkpoint.

For gauge alignment study. Same scaffold as ace_retarget but saves z_b instead
of decoded motion. Works for AnchorLabelFlow (and likely AL-Flow-Src / AL-Flow-Src-G
with minor adaptation if their generator interfaces differ).

Usage:
    python /tmp/save_alflow_latents.py \
        --ckpt save/actionbridge/anchor_label_flow_v1/ckpt_final.pt \
        --manifest eval/benchmark_v3/queries_sif_intersection/manifest.json \
        --out_dir /tmp/alflow_latents/v1
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

from data_loaders.truebones.truebones_utils.param_utils import OBJECT_SUBSETS_DICT, DATASET_DIR
from eval.benchmark_v3.action_taxonomy import (
    ACTION_CLUSTERS, parse_action_from_filename, action_to_cluster,
)
from model.actionbridge.generator_anchor import AnchorLabelFlowGenerator
from model.moreflow.skel_graph import (
    SkelGraphEncoder, build_skel_features, pad_to_max_joints,
)
from model.moreflow.stage_a_registry import StageARegistry

CLUSTERS = sorted(ACTION_CLUSTERS.keys())
CLUSTER_TO_IDX = {c: i for i, c in enumerate(CLUSTERS)}

# Try to mirror exact-action mapping used by AL-Flow (per generator_anchor.py)
EXACT_ACTIONS_FILE = PROJECT_ROOT / 'eval/benchmark_v3/exact_action_idx.json'
EXACT_ACTION_TO_IDX = {}
if EXACT_ACTIONS_FILE.exists():
    EXACT_ACTION_TO_IDX = json.loads(EXACT_ACTIONS_FILE.read_text())


def sample_z_b(G, z_init, q_t, tgt_id, tgt_graph, cluster_id, exact_id,
               n_steps=1, cfg_scale=1.0):
    """Single-step Euler: z_b = z_init + v(z_init, t=0). Mirrors run_anchor_label_flow_v5."""
    z = z_init
    if n_steps == 1:
        t = torch.zeros(z.shape[0], device=z.device)
        v_cond = G(z, t, tgt_id, tgt_graph, cluster_id, exact_id)
        if cfg_scale != 1.0:
            cmask = torch.ones_like(cluster_id, dtype=torch.bool)
            emask = torch.ones_like(exact_id, dtype=torch.bool)
            v_uncond = G(z, t, tgt_id, tgt_graph, cluster_id, exact_id,
                         cluster_mask=cmask, exact_mask=emask)
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v = v_cond
        return z + v
    # Multi-step Euler
    dt = 1.0 / n_steps
    for k in range(n_steps):
        t = torch.full((z.shape[0],), float(k) * dt, device=z.device)
        v_cond = G(z, t, tgt_id, tgt_graph, cluster_id, exact_id)
        if cfg_scale != 1.0:
            cmask = torch.ones_like(cluster_id, dtype=torch.bool)
            emask = torch.ones_like(exact_id, dtype=torch.bool)
            v_uncond = G(z, t, tgt_id, tgt_graph, cluster_id, exact_id,
                         cluster_mask=cmask, exact_mask=emask)
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v = v_cond
        z = z + dt * v
    return z


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--manifest', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--n_euler_steps', type=int, default=1)
    parser.add_argument('--cfg_scale', type=float, default=1.0)
    parser.add_argument('--seed_for_z_init', type=int, default=42,
                        help='Fix z_init RNG so all checkpoints see the same noise')
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[alflow-latents] Device: {device}")
    print(f"[alflow-latents] Loading ckpt: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    skels = ckpt['skels']
    skel_to_id = ckpt['skel_to_id']

    G = AnchorLabelFlowGenerator(
        n_skels=len(skels),
        n_clusters=len(CLUSTERS) + 1,
        n_exact_actions=ckpt.get('n_exact_actions', (max(EXACT_ACTION_TO_IDX.values())+1 if EXACT_ACTION_TO_IDX else 124) + 1),
    ).to(device)
    G.load_state_dict(ckpt['G'])
    G.eval()
    skel_enc = SkelGraphEncoder().to(device)
    skel_enc.load_state_dict(ckpt['skel_enc'])
    skel_enc.eval()

    print("[alflow-latents] Loading Stage A registry for ALL 70 skels...")
    all_skels = list(set(OBJECT_SUBSETS_DICT['train_v3']) | set(OBJECT_SUBSETS_DICT['test_v3']))
    registry = StageARegistry(all_skels, device=device)
    cond_dict = np.load(Path(DATASET_DIR) / 'cond.npy', allow_pickle=True).item()
    skel_features = build_skel_features(cond_dict)
    max_J = max(skel_features[s]['n_joints'] for s in skels)

    print(f"[alflow-latents] Loading manifest: {args.manifest}")
    manifest = json.loads(Path(args.manifest).read_text())
    queries = manifest.get('queries', manifest.get('triples', []))
    if args.limit:
        queries = queries[:args.limit]
    print(f"[alflow-latents] Processing {len(queries)} queries...")

    # FIX z_init RNG so two seeds see the same noise → fair gauge comparison
    rng = np.random.RandomState(args.seed_for_z_init)
    triple_meta = []
    t0 = time.time()
    with torch.no_grad():
        for i, q in enumerate(queries):
            try:
                tgt_skel = q['skel_b']
                src_action = q.get('src_action', '')
                cluster_str = action_to_cluster(parse_action_from_filename(q.get('src_fname', ''))) or ''
                cluster_id_int = CLUSTER_TO_IDX.get(cluster_str, 0)
                exact_id_int = EXACT_ACTION_TO_IDX.get(src_action, 0)

                # Sequence length: AL-Flow generator is hard-coded to T=8 tokens (positional encoding fixed)
                n_token = 8

                codebook_dim = registry.get(tgt_skel)['model'].codebook_dim
                z_init_np = rng.randn(1, n_token, codebook_dim).astype(np.float32)
                z_init = torch.from_numpy(z_init_np).to(device)

                tgt_id = torch.full((1,), skel_to_id.get(tgt_skel, 0), dtype=torch.long, device=device)
                tgt_pj_pad, tgt_pj_mask = pad_to_max_joints(skel_features[tgt_skel]['per_joint'], max_J)
                tgt_pj_pad = tgt_pj_pad.unsqueeze(0).to(device)
                tgt_pj_mask = tgt_pj_mask.unsqueeze(0).to(device)
                tgt_a = skel_features[tgt_skel]['agg'].unsqueeze(0).to(device)
                tgt_graph = skel_enc(tgt_pj_pad, tgt_pj_mask, tgt_a)
                cid = torch.full((1,), cluster_id_int, dtype=torch.long, device=device)
                eid = torch.full((1,), exact_id_int, dtype=torch.long, device=device)

                z_b = sample_z_b(G, z_init, q_t=None, tgt_id=tgt_id, tgt_graph=tgt_graph,
                                 cluster_id=cid, exact_id=eid,
                                 n_steps=args.n_euler_steps, cfg_scale=args.cfg_scale)
                z_lat = z_b.squeeze(0).cpu().numpy()  # [n_token, codebook_dim]
                np.save(out_dir / f'z_query_{i:04d}.npy', z_lat)
                triple_meta.append({
                    'query_id': i, 'src_skel': q.get('skel_a'), 'tgt_skel': tgt_skel,
                    'src_fname': q.get('src_fname'), 'T_tokens': z_lat.shape[0],
                    'codebook_dim': z_lat.shape[1],
                    'cluster_id': cluster_id_int, 'exact_id': exact_id_int,
                })
            except Exception as e:
                print(f"  [{i+1}/{len(queries)}] FAILED: {type(e).__name__}: {e}")
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(queries)}] done ({time.time() - t0:.0f}s)")

    (out_dir / 'meta.json').write_text(json.dumps(triple_meta, indent=2))
    print(f"[alflow-latents] saved {len(triple_meta)} queries to {out_dir} in {time.time() - t0:.0f}s")


if __name__ == '__main__':
    main()
