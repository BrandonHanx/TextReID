import copy
import math
import random
from collections import defaultdict

import torch
from torch.utils.data.sampler import BatchSampler


def _split(tensor, size, dim=0, drop_last=False):
    if dim < 0:
        dim += tensor.dim()
    dim_size = tensor.size(dim)

    if dim_size < size:
        times = math.ceil(size / dim_size)
        tensor = tensor.repeat_interleave(times)
        dim_size = size

    split_size = size
    num_splits = (dim_size + split_size - 1) // split_size
    last_split_size = split_size - (split_size * num_splits - dim_size)

    def get_split_size(i):
        return split_size if i < num_splits - 1 else last_split_size

    if drop_last and last_split_size != split_size:
        total_num_splits = num_splits - 1
    else:
        total_num_splits = num_splits

    return list(
        tensor.narrow(int(dim), int(i * split_size), int(get_split_size(i)))
        for i in range(0, total_num_splits)
    )


def _merge(splits, pids, num_pids_per_batch):
    avaible_pids = copy.deepcopy(pids)
    merged = []

    while len(avaible_pids) >= num_pids_per_batch:
        batch = []
        selected_pids = random.sample(avaible_pids, num_pids_per_batch)
        for pid in selected_pids:
            batch_idxs = splits[pid].pop(0)
            batch.extend(batch_idxs.tolist())
            if len(splits[pid]) == 0:
                avaible_pids.remove(pid)
        merged.append(batch)
    return merged


def _map(dataset):
    id_to_img_map = []
    for i in range(len(dataset)):
        _, pid = dataset.get_id_info(i)
        id_to_img_map.append(pid)
    return id_to_img_map


class TripletSampler(BatchSampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, sampler, data_source, batch_size, images_per_pid, drop_last):
        super(TripletSampler, self).__init__(sampler, batch_size, drop_last)
        self.num_instances = images_per_pid
        self.num_pids_per_batch = batch_size // images_per_pid
        self.id_to_img_map = _map(data_source)
        self.index_dict = defaultdict(list)
        for index, pid in enumerate(self.id_to_img_map):
            self.index_dict[pid].append(index)
        self.pids = list(self.index_dict.keys())

        self.group_ids = torch.as_tensor(self.id_to_img_map)
        self.groups = torch.unique(self.group_ids).sort(0)[0]

        self._can_reuse_batches = False

    def _prepare_batches(self):
        dataset_size = len(self.group_ids)
        sampled_ids = torch.as_tensor(list(self.sampler))
        order = torch.full((dataset_size,), -1, dtype=torch.int64)
        order[sampled_ids] = torch.arange(len(sampled_ids))

        mask = order >= 0
        clusters = [(self.group_ids == i) & mask for i in self.groups]
        relative_order = [order[cluster] for cluster in clusters]
        permutation_ids = [s.sort()[0] for s in relative_order]
        permuted_clusters = [sampled_ids[idx] for idx in permutation_ids]

        splits = defaultdict(list)
        for idx, c in enumerate(permuted_clusters):
            splits[idx] = _split(c, self.num_instances, drop_last=True)
        merged = _merge(splits, self.pids, self.num_pids_per_batch)

        first_element_of_batch = [t[0] for t in merged]
        inv_sampled_ids_map = {v: k for k, v in enumerate(sampled_ids.tolist())}
        first_index_of_batch = torch.as_tensor(
            [inv_sampled_ids_map[s] for s in first_element_of_batch]
        )
        permutation_order = first_index_of_batch.sort(0)[1].tolist()
        batches = [merged[i] for i in permutation_order]

        return batches

    def __iter__(self):
        if self._can_reuse_batches:
            batches = self._batches
            self._can_reuse_batches = False
        else:
            batches = self._prepare_batches()
        self._batches = batches

        for batch in iter(batches):
            yield batch

    def __len__(self):
        if not hasattr(self, "_batches"):
            self._batches = self._prepare_batches()
            self._can_reuse_batches = True
        return len(self._batches)
