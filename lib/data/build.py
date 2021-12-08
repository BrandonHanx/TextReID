import torch.utils.data

from lib.config.paths_catalog import DatasetCatalog
from lib.utils.comm import get_world_size

from . import datasets as D
from . import samplers
from .collate_batch import collate_fn
from .transforms import build_transforms


def build_dataset(cfg, dataset_list, transforms, dataset_catalog, is_train=True):
    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError(
            "dataset_list should be a list of strings, got {}".format(dataset_list)
        )
    datasets = []
    for dataset_name in dataset_list:
        data = dataset_catalog.get(cfg.ROOT, dataset_name)
        factory = getattr(D, data["factory"])
        args = data["args"]
        args["transforms"] = transforms

        if data["factory"] == "CUHKPEDESDataset":
            args["use_onehot"] = cfg.DATASETS.USE_ONEHOT
            args["max_length"] = 105

        # make dataset from factory
        dataset = factory(**args)
        datasets.append(dataset)

    # for testing, return a list of datasets
    if not is_train:
        return datasets

    # for training, concatenate all datasets into a single one
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = D.ConcatDataset(datasets)

    return [dataset]


def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return torch.utils.data.distributed.DistributedSampler(dataset)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def make_batch_data_sampler(cfg, dataset, sampler, images_per_batch, is_train=True):
    if is_train and cfg.DATALOADER.EN_SAMPLER:
        batch_sampler = samplers.TripletSampler(
            sampler,
            dataset,
            images_per_batch,
            cfg.DATALOADER.IMS_PER_ID,
            drop_last=True,
        )
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, images_per_batch, drop_last=is_train
        )
    return batch_sampler


def make_data_loader(cfg, is_train=True, is_distributed=False):
    num_gpus = get_world_size()
    if is_train:
        images_per_batch = cfg.SOLVER.IMS_PER_BATCH
        assert (
            images_per_batch % num_gpus == 0
        ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(
            images_per_batch, num_gpus
        )
        images_per_gpu = images_per_batch // num_gpus
        shuffle = True
    else:
        images_per_batch = cfg.TEST.IMS_PER_BATCH
        assert (
            images_per_batch % num_gpus == 0
        ), "TEST.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(
            images_per_batch, num_gpus
        )
        images_per_gpu = images_per_batch // num_gpus
        shuffle = is_distributed

    dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST

    transforms = build_transforms(cfg, is_train)

    datasets = build_dataset(cfg, dataset_list, transforms, DatasetCatalog, is_train)

    data_loaders = []
    for dataset in datasets:
        sampler = make_data_sampler(dataset, shuffle, is_distributed)
        batch_sampler = make_batch_data_sampler(
            cfg, dataset, sampler, images_per_gpu, is_train
        )
        num_workers = cfg.DATALOADER.NUM_WORKERS
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
        )
        data_loaders.append(data_loader)
    if is_train:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1
        return data_loaders[0]
    return data_loaders
