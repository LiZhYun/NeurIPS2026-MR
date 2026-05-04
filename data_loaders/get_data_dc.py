"""Dataloader builder for D+C training."""
from torch.utils.data import DataLoader
from data_loaders.tensors_dc import truebones_batch_collate_dc
from data_loaders.truebones.data.dataset_dc import TruebonesDC


def get_dataset_loader_dc(batch_size, num_frames, split='train',
                          temporal_window=31, t5_name='t5-base',
                          balanced=True, objects_subset='all',
                          split_files=None,
                          aug_prob_noop=0.75, aug_prob_remove=0.125,
                          aug_prob_add=0.125):
    dataset = TruebonesDC(
        split=split, num_frames=num_frames, temporal_window=temporal_window,
        t5_name=t5_name, balanced=balanced, objects_subset=objects_subset,
        split_files=split_files,
        aug_prob_noop=aug_prob_noop,
        aug_prob_remove=aug_prob_remove,
        aug_prob_add=aug_prob_add,
    )

    sampler = None
    if balanced:
        from data_loaders.truebones.data.dataset import TruebonesSampler
        sampler = TruebonesSampler(dataset)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=8,
        drop_last=True,
        collate_fn=truebones_batch_collate_dc,
    )
    return loader
