import numpy as np
import random
from os.path import join as pjoin

from data_loaders.truebones.data.dataset import MotionDataset, Truebones
from data_loaders.truebones.truebones_utils.get_opt import get_opt


class MotionDatasetConditioned(MotionDataset):
    """Extends MotionDataset to also return pre-augmentation source motion.

    Items 0–13 are identical to MotionDataset.__getitem__.
    Item 14: source_motion_norm [T, J_src, 13] — original joints, same temporal crop
    Item 15: source_offsets     [J_src, 3]     — original rest-pose bone offsets
    """
    def __getitem__(self, item):
        idx  = item if self.balanced else self.pointer + item
        data = self.data_dict[self.name_list[idx]]

        # Capture pre-augmentation source motion
        object_type       = data['object_type']
        source_mean       = self.cond_dict[object_type]['mean']
        source_std        = self.cond_dict[object_type]['std'] + 1e-6
        source_motion_raw = data['motion'].copy()   # [T, J_src, 13]
        source_offsets    = data['offsets'].copy()   # [J_src, 3]
        source_motion_norm = (source_motion_raw - source_mean[None, :]) / source_std[None, :]
        source_motion_norm = np.nan_to_num(source_motion_norm)

        # Augmented motion (may have different joint count, same frame count)
        motion, m_length, object_type, parents, joints_graph_dist, joints_relations, \
            tpos_first_frame, offsets, joints_names_embs, kinematic_chains, mean, std = self.augment(data)

        std   += 1e-6
        motion = (motion - mean[None, :]) / std[None, :]
        motion = np.nan_to_num(motion)
        tpos_first_frame = (tpos_first_frame - mean) / std
        tpos_first_frame = np.nan_to_num(tpos_first_frame)

        ind = 0
        if m_length < self.max_motion_length:
            pad = self.max_motion_length - m_length
            motion = np.concatenate(
                [motion, np.zeros((pad, motion.shape[1], motion.shape[2]))], axis=0)
            source_motion_norm = np.concatenate(
                [source_motion_norm, np.zeros((pad, source_motion_norm.shape[1], 13))], axis=0)
        elif m_length > self.max_motion_length:
            ind = random.randint(0, m_length - self.max_motion_length)
            motion             = motion[ind: ind + self.max_motion_length]
            source_motion_norm = source_motion_norm[ind: ind + self.max_motion_length]
            m_length           = self.max_motion_length

        motion_name = self.name_list[idx]  # e.g. "Horse___SlowWalk_432"
        return (motion, m_length, parents, tpos_first_frame, offsets,
                self.temporal_mask_template, joints_graph_dist, joints_relations,
                object_type, joints_names_embs, ind, mean, std, self.opt.max_joints,
                source_motion_norm,   # [T, J_src, 13]
                source_offsets,       # [J_src, 3]
                motion_name)          # str


class TruebonesConditioned(Truebones):
    """Truebones dataset variant that returns conditioned (source+target) samples."""

    def __init__(self, split="train", temporal_window=31, t5_name='t5-base', split_files=None, **kwargs):
        # Replicate Truebones.__init__ but use MotionDatasetConditioned
        print("in TruebonesConditioned constructor")
        abs_base_path = '.'
        opt = get_opt(None)
        opt.motion_dir  = pjoin(abs_base_path, opt.motion_dir)
        opt.data_root   = pjoin(abs_base_path, opt.data_root)
        opt.max_motion_length = min(opt.max_motion_length, kwargs['num_frames'])
        self.opt             = opt
        self.balanced        = kwargs['balanced']
        self.objects_subset  = kwargs.get('objects_subset', 'all')

        import numpy as np
        cond_dict = np.load(opt.cond_file, allow_pickle=True).item()
        subset    = opt.subsets_dict[self.objects_subset]
        cond_dict = {k: cond_dict[k] for k in subset if k in cond_dict}
        print(f'TruebonesConditioned: {len(cond_dict)} characters in subset "{self.objects_subset}"')

        self.split_file   = pjoin(opt.data_root, f'{split}.txt')
        self.motion_dataset = MotionDatasetConditioned(
            opt, cond_dict, temporal_window, t5_name, self.balanced, split_files=split_files)
        assert len(self.motion_dataset) > 1, 'Dataset is empty — check data directory.'
