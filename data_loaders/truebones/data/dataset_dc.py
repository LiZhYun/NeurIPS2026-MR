"""Dataset wrapper for D+C training.

Extends MotionDatasetConditioned so we can control the augmentation probability
(to hit the reviewer-locked 75% self / 25% paired ratio on average) and expose
the actual aug_type sampled per item, so the training loop can split losses
cleanly.
"""
import random
import numpy as np
from os.path import join as pjoin

from data_loaders.truebones.data.dataset_conditioned import MotionDatasetConditioned, TruebonesConditioned
from data_loaders.truebones.truebones_utils.get_opt import get_opt
from data_loaders.truebones.truebones_utils.motion_process import (
    remove_joints_augmentation,
    add_joint_augmentation,
)


class MotionDatasetDC(MotionDatasetConditioned):
    """Adds:
      - configurable probabilities for [no-aug, remove, add]
      - returns aug_type as part of each item so the training loop can route
        samples into L_self (aug=0) vs L_pair (aug!=0)
    """
    def __init__(self, *args, aug_prob_noop=0.75, aug_prob_remove=0.125,
                 aug_prob_add=0.125, **kwargs):
        super().__init__(*args, **kwargs)
        s = aug_prob_noop + aug_prob_remove + aug_prob_add
        assert abs(s - 1.0) < 1e-6, f"probs must sum to 1, got {s}"
        self.aug_prob_noop = aug_prob_noop
        self.aug_prob_remove = aug_prob_remove
        self.aug_prob_add = aug_prob_add

    def _sample_aug_type(self, object_type):
        r = random.random()
        if r < self.aug_prob_noop:
            return 0
        if r < self.aug_prob_noop + self.aug_prob_remove:
            return 1 if object_type != "Dragon" else 0
        return 2

    def augment(self, data, aug_type_override=None):
        object_type = data['object_type']
        mean = self.cond_dict[object_type]['mean']
        std = self.cond_dict[object_type]['std']
        aug_type = aug_type_override if aug_type_override is not None else \
            self._sample_aug_type(object_type)
        if aug_type == 0:
            return (data['motion'], data['length'], data['object_type'],
                    data['parents'], data['joints_graph_dist'],
                    data['joints_relations'], data['tpos_first_frame'],
                    data['offsets'], data['joints_names_embs'],
                    data['kinematic_chains'], mean, std), 0
        if aug_type == 1:
            removal_rate = random.choice([0.1, 0.2, 0.3])
            return remove_joints_augmentation(data, removal_rate, mean, std), 1
        return add_joint_augmentation(data, mean, std), 2

    def __getitem__(self, item):
        idx = item if self.balanced else self.pointer + item
        data = self.data_dict[self.name_list[idx]]

        object_type = data['object_type']
        source_mean = self.cond_dict[object_type]['mean']
        source_std = self.cond_dict[object_type]['std'] + 1e-6
        source_motion_raw = data['motion'].copy()
        source_offsets = data['offsets'].copy()
        source_motion_norm = (source_motion_raw - source_mean[None, :]) / source_std[None, :]
        source_motion_norm = np.nan_to_num(source_motion_norm)

        aug_tuple, aug_type = self.augment(data)
        (motion, m_length, object_type, parents, joints_graph_dist,
         joints_relations, tpos_first_frame, offsets, joints_names_embs,
         kinematic_chains, mean, std) = aug_tuple

        std = std + 1e-6
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
                [source_motion_norm,
                 np.zeros((pad, source_motion_norm.shape[1], 13))], axis=0)
        elif m_length > self.max_motion_length:
            ind = random.randint(0, m_length - self.max_motion_length)
            motion = motion[ind: ind + self.max_motion_length]
            source_motion_norm = source_motion_norm[ind: ind + self.max_motion_length]
            m_length = self.max_motion_length

        motion_name = self.name_list[idx]
        return (motion, m_length, parents, tpos_first_frame, offsets,
                self.temporal_mask_template, joints_graph_dist, joints_relations,
                object_type, joints_names_embs, ind, mean, std, self.opt.max_joints,
                source_motion_norm, source_offsets, motion_name,
                aug_type)  # NEW


class TruebonesDC(TruebonesConditioned):
    """Same as TruebonesConditioned but uses MotionDatasetDC under the hood."""
    def __init__(self, split="train", temporal_window=31, t5_name='t5-base',
                 split_files=None, aug_prob_noop=0.75, aug_prob_remove=0.125,
                 aug_prob_add=0.125, **kwargs):
        print("in TruebonesDC constructor")
        abs_base_path = '.'
        opt = get_opt(None)
        opt.motion_dir = pjoin(abs_base_path, opt.motion_dir)
        opt.data_root = pjoin(abs_base_path, opt.data_root)
        opt.max_motion_length = min(opt.max_motion_length, kwargs['num_frames'])
        self.opt = opt
        self.balanced = kwargs['balanced']
        self.objects_subset = kwargs.get('objects_subset', 'all')

        cond_dict = np.load(opt.cond_file, allow_pickle=True).item()
        subset = opt.subsets_dict[self.objects_subset]
        cond_dict = {k: cond_dict[k] for k in subset if k in cond_dict}
        print(f'TruebonesDC: {len(cond_dict)} characters in subset "{self.objects_subset}"')

        self.split_file = pjoin(opt.data_root, f'{split}.txt')
        self.motion_dataset = MotionDatasetDC(
            opt, cond_dict, temporal_window, t5_name, self.balanced,
            split_files=split_files,
            aug_prob_noop=aug_prob_noop,
            aug_prob_remove=aug_prob_remove,
            aug_prob_add=aug_prob_add,
        )
        assert len(self.motion_dataset) > 1, 'Dataset is empty — check data directory.'
