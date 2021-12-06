import datetime
import logging
import os
import time
from collections import defaultdict

import torch
from tqdm import tqdm

from lib.data.metrics import evaluation
from lib.utils.comm import all_gather, is_main_process, synchronize


def compute_on_dataset(model, data_loader, device):
    model.eval()
    results_dict = defaultdict(list)
    for batch in tqdm(data_loader):
        images, captions, image_ids = batch
        images = images.to(device)
        captions = [caption.to(device) for caption in captions]
        with torch.no_grad():
            output = model(images, captions)
        for result in output:
            for img_id, pred in zip(image_ids, result):
                results_dict[img_id].append(pred)
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("PersonSearch.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )
    return predictions


def inference(
    model,
    data_loader,
    dataset_name="cuhkpedes-test",
    device="cuda",
    output_folder="",
    save_data=True,
    rerank=True,
):
    logger = logging.getLogger("PersonSearch.inference")
    dataset = data_loader.dataset
    logger.info(
        "Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset))
    )

    predictions = None
    if not os.path.exists(os.path.join(output_folder, "inference_data.npz")):
        # convert to a torch.device for efficiency
        device = torch.device(device)
        num_devices = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )
        start_time = time.time()

        predictions = compute_on_dataset(model, data_loader, device)
        # wait for all processes to complete before measuring the time
        synchronize()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=total_time))
        logger.info(
            "Total inference time: {} ({} s / img per device, on {} devices)".format(
                total_time_str, total_time * num_devices / len(dataset), num_devices
            )
        )
        predictions = _accumulate_predictions_from_multiple_gpus(predictions)

        if not is_main_process():
            return

    return evaluation(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        save_data=save_data,
        rerank=rerank,
        topk=[1, 5, 10],
    )
