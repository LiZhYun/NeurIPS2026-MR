import torch
from data_loaders.tensors import (
    truebones_collate, collate_tensors, create_padded_relation
)


def truebones_batch_collate_conditioned(batch):
    """Extends truebones_batch_collate to also pack source motion into cond['y'].

    Input batch items are 16-tuples:
      0–13: same as truebones_batch_collate
      14: source_motion_norm [T, J_src, 13]
      15: source_offsets     [J_src, 3]
    """
    max_joints = batch[0][13]
    adapted_batch = []

    for b in batch:
        max_len, n_joints, n_feats = b[0].shape

        # --- Standard fields (identical to truebones_batch_collate) ---
        tpos_first_frame = torch.zeros((max_joints, n_feats))
        tpos_first_frame[:n_joints] = torch.tensor(b[3])

        motion = torch.zeros((max_len, max_joints, n_feats))
        motion[:, :n_joints, :] = torch.tensor(b[0])

        joints_names_embs = torch.zeros((max_joints, b[9].shape[1]))
        joints_names_embs[:n_joints] = torch.tensor(b[9])

        mean = torch.zeros((max_joints, n_feats))
        mean[:n_joints] = torch.tensor(b[11])

        std = torch.ones((max_joints, n_feats))
        std[:n_joints] = torch.tensor(b[12])

        temporal_mask          = b[5][:max_len + 1, :max_len + 1].clone()
        padded_joints_relations = create_padded_relation(b[7], max_joints, n_joints)
        padded_graph_dist       = create_padded_relation(b[6], max_joints, n_joints)

        # --- Source motion: item[14] = [T, J_src, 13] ---
        src_motion_raw = b[14]                             # [T, J_src, 13]
        n_joints_src   = src_motion_raw.shape[1]
        source_motion  = torch.zeros((max_len, max_joints, n_feats))
        source_motion[:, :n_joints_src, :] = torch.tensor(src_motion_raw)
        source_motion  = source_motion.permute(1, 2, 0).float()   # [J_max, 13, T]

        # --- Source offsets: item[15] = [J_src, 3] ---
        source_offsets = torch.zeros((max_joints, 3))
        source_offsets[:n_joints_src] = torch.tensor(b[15])

        # --- Source joints mask ---
        source_joints_mask = torch.zeros(max_joints, dtype=torch.bool)
        source_joints_mask[:n_joints_src] = True

        item = {
            'inp':              motion.permute(1, 2, 0).float(),   # [J, 13, T]
            'n_joints':         n_joints,
            'lengths':          b[1],
            'parents':          b[2],
            'temporal_mask':    temporal_mask,
            'graph_dist':       padded_graph_dist,
            'joints_relations': padded_joints_relations,
            'object_type':      b[8],
            'joints_names_embs': joints_names_embs,
            'tpos_first_frame': tpos_first_frame,
            'crop_start_ind':   b[10],
            'mean':             mean,
            'std':              std,
            # source fields (not consumed by truebones_collate, handled below)
            'source_motion':       source_motion,
            'source_offsets':      source_offsets,
            'source_joints_mask':  source_joints_mask,
            'source_name':         b[16] if len(b) > 16 else b[8],  # motion name or object_type fallback
        }
        adapted_batch.append(item)

    motion, cond = truebones_collate(adapted_batch)

    cond['y']['source_motion']      = collate_tensors([b['source_motion']      for b in adapted_batch])
    cond['y']['source_offsets']     = collate_tensors([b['source_offsets']     for b in adapted_batch])
    cond['y']['source_joints_mask'] = torch.stack(   [b['source_joints_mask'] for b in adapted_batch])
    cond['y']['source_name']        = [b.get('source_name', '') for b in adapted_batch]

    return motion, cond
