from torch.utils.data import DataLoader
from data_loaders.tensors_conditioned import truebones_batch_collate_conditioned
from data_loaders.truebones.data.dataset_conditioned import TruebonesConditioned


def get_dataset_loader_conditioned(batch_size, num_frames, split='train',
                                   temporal_window=31, t5_name='t5-base',
                                   balanced=True, objects_subset='all', split_files=None):
    dataset = TruebonesConditioned(
        split=split, num_frames=num_frames, temporal_window=temporal_window,
        t5_name=t5_name, balanced=balanced, objects_subset=objects_subset,
        split_files=split_files)

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
        collate_fn=truebones_batch_collate_conditioned,
    )
    return loader
