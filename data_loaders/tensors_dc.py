"""Collate for D+C. Same as conditioned but propagates the aug_type per sample."""
import torch
from data_loaders.tensors import (
    truebones_collate, collate_tensors, create_padded_relation
)


def truebones_batch_collate_dc(batch):
    max_joints = batch[0][13]
    adapted_batch = []
    aug_types = []

    for b in batch:
        max_len, n_joints, n_feats = b[0].shape

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

        temporal_mask = b[5][:max_len + 1, :max_len + 1].clone()
        padded_joints_relations = create_padded_relation(b[7], max_joints, n_joints)
        padded_graph_dist = create_padded_relation(b[6], max_joints, n_joints)

        src_motion_raw = b[14]
        n_joints_src = src_motion_raw.shape[1]
        source_motion = torch.zeros((max_len, max_joints, n_feats))
        source_motion[:, :n_joints_src, :] = torch.tensor(src_motion_raw)
        source_motion = source_motion.permute(1, 2, 0).float()

        source_offsets = torch.zeros((max_joints, 3))
        source_offsets[:n_joints_src] = torch.tensor(b[15])

        source_joints_mask = torch.zeros(max_joints, dtype=torch.bool)
        source_joints_mask[:n_joints_src] = True

        item = {
            'inp': motion.permute(1, 2, 0).float(),
            'n_joints': n_joints,
            'lengths': b[1],
            'parents': b[2],
            'temporal_mask': temporal_mask,
            'graph_dist': padded_graph_dist,
            'joints_relations': padded_joints_relations,
            'object_type': b[8],
            'joints_names_embs': joints_names_embs,
            'tpos_first_frame': tpos_first_frame,
            'crop_start_ind': b[10],
            'mean': mean,
            'std': std,
            'source_motion': source_motion,
            'source_offsets': source_offsets,
            'source_joints_mask': source_joints_mask,
            'source_name': b[16],
        }
        adapted_batch.append(item)
        aug_types.append(int(b[17]))

    motion, cond = truebones_collate(adapted_batch)
    cond['y']['source_motion'] = collate_tensors([b['source_motion'] for b in adapted_batch])
    cond['y']['source_offsets'] = collate_tensors([b['source_offsets'] for b in adapted_batch])
    cond['y']['source_joints_mask'] = torch.stack([b['source_joints_mask'] for b in adapted_batch])
    cond['y']['source_name'] = [b['source_name'] for b in adapted_batch]
    cond['y']['aug_type'] = torch.tensor(aug_types, dtype=torch.long)
    return motion, cond
