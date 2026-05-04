import torch
from pytorch3d.transforms import rotation_6d_to_matrix, so3_relative_angle # pip install pytorch3d

def avg_per_frame_dist(motion1, motion2, norm):
    # [n_frames, n_features]
    min_len = min(motion1.shape[0], motion2.shape[0])
    if norm == 'fro':
        n_frames = motion1.shape[0]
        return torch.norm(motion2[:min_len] - motion1[:min_len], p='fro').cpu().numpy() / n_frames
    elif norm == 'l2':
        return torch.norm(motion2[:min_len] - motion1[:min_len], p=2, dim=-1).mean().cpu().numpy()
    elif norm == 'loc':
        n_joints = motion1.shape[-1] // 3
        return torch.norm(motion2[:min_len].view(-1, n_joints, 3) - motion1[:min_len].view(-1, n_joints, 3), p=2, dim=-1).mean().cpu().numpy()
    elif norm == 'rot':
        n_joints = motion1.shape[-1] // 6
        motion1_matrices = rotation_6d_to_matrix(motion1[:min_len].view(min_len, n_joints, 6)).view(-1, 3, 3)
        motion2_matrices = rotation_6d_to_matrix(motion2[:min_len].view(min_len, n_joints, 6)).view(-1, 3, 3)
        # Compute relative angles for each joint
        angles = so3_relative_angle(
            motion1_matrices,  # [n_frames1, n_frames2, n_joints, 3, 3]
            motion2_matrices,  # [n_frames1, n_frames2, n_joints, 3, 3]
            cos_angle=False
        ).view(min_len, n_joints)  # [n_frames1, n_frames2, n_joints]
        return angles.mean()
    else:
        raise ValueError(f'invalid nort type [{norm}]')

def pos_avg_l2(motion1, motion2):
    n_joints = motion1.shape[-1] // 3
    return torch.norm(motion2.view(*motion2.shape[:2], n_joints, 3) - motion1.view(*motion1.shape[:2], n_joints, 3), p=2, dim=-1).mean(dim=-1).cpu().numpy()

def pos_avg_cosine_distance(motion1, motion2, chunk_size=128):
    """
    Compute the angular distance matrix between two motions represented in 6D format.
    Uses chunked computation to avoid multi-GB memory allocations.

    Args:
        motion1: Tensor of shape [n_frames1, 1, n_joints * 6].
        motion2: Tensor of shape [1, n_frames2, n_joints * 6].

    Returns:
        dist: numpy array of shape [n_frames1, n_frames2].
    """
    n_joints = motion1.shape[-1] // 6
    n1 = motion1.shape[0]
    n2 = motion2.shape[1]

    motion1 = motion1.view(n1, 1, n_joints, 6).cuda()
    motion2 = motion2.view(1, n2, n_joints, 6).cuda()

    m1_mat = rotation_6d_to_matrix(motion1)  # [n1, 1, n_joints, 3, 3]

    dist = torch.empty(n1, n2)
    for j in range(0, n2, chunk_size):
        j_end = min(j + chunk_size, n2)
        m2_chunk = rotation_6d_to_matrix(motion2[:, j:j_end])  # [1, chunk, n_joints, 3, 3]
        cs = j_end - j
        a = m1_mat.expand(n1, cs, n_joints, 3, 3).reshape(-1, 3, 3)
        b = m2_chunk.expand(n1, cs, n_joints, 3, 3).reshape(-1, 3, 3)
        angles = so3_relative_angle(a, b, cos_angle=False).view(n1, cs, n_joints)
        dist[:, j:j_end] = angles.mean(dim=-1).cpu()

    return dist