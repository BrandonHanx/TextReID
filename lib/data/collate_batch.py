import torch


def collate_fn(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0])
    captions = transposed_batch[1]
    img_ids = transposed_batch[2]
    return images, captions, img_ids
