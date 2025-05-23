from torch.utils.data import DataLoader
from load.tensors import collate as all_collate


def get_dataset_class(name):
    if name == 'karate':
        from .data_loaders.karate import KaratePoses
        return KaratePoses
    elif name == 'humanact12':
        from .data_loaders.humanact12poses import HumanAct12Poses
        return HumanAct12Poses
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')


def get_collate_fn():
    return all_collate


def get_dataset(name, num_frames, test_participant, pose_rep, split='train', data_array=None):
    data = get_dataset_class(name)
    dataset = data(
        test_participant=test_participant,
        split=split,
        num_frames=num_frames,
        pose_rep=pose_rep,
        data_array=data_array
    )
    return dataset


def get_dataset_loader(name, batch_size, num_frames, test_participant, pose_rep, split='train', data_array=None):
    dataset = get_dataset(name, num_frames, test_participant, pose_rep, split, data_array)
    collate = get_collate_fn()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, drop_last=False, collate_fn=collate
    )
    return loader
