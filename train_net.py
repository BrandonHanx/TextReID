import argparse
import os
import random

import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from lib.config import cfg
from lib.data import make_data_loader
from lib.engine.trainer import do_train
from lib.models.model import build_model
from lib.solver import make_lr_scheduler, make_optimizer
from lib.utils.checkpoint import Checkpointer
from lib.utils.comm import get_rank, synchronize
from lib.utils.directory import makedir
from lib.utils.logger import setup_logger
from lib.utils.metric_logger import MetricLogger, TensorboardLogger


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(cfg, output_dir, local_rank, distributed, resume_from, use_tensorboard):
    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
    )
    data_loader_val = make_data_loader(
        cfg,
        is_train=False,
        is_distributed=distributed,
    )
    model = build_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )

    arguments = {}
    arguments["iteration"] = 0
    arguments["epoch"] = 0

    save_to_disk = get_rank() == 0
    checkpointer = Checkpointer(model, optimizer, scheduler, output_dir, save_to_disk)
    if cfg.MODEL.WEIGHT != "imagenet":
        if os.path.isfile(cfg.MODEL.WEIGHT):
            checkpointer.load(cfg.MODEL.WEIGHT)
        else:
            raise IOError("{} is not a checkpoint file".format(cfg.MODEL.WEIGHT))
    if resume_from:
        if os.path.isfile(resume_from):
            extra_checkpoint_data = checkpointer.resume(resume_from)
            arguments.update(extra_checkpoint_data)
        else:
            raise IOError("{} is not a checkpoint file".format(resume_from))

    if use_tensorboard:
        meters = TensorboardLogger(
            log_dir=os.path.join(output_dir, "tensorboard"),
            start_iter=arguments["iteration"],
            delimiter="  ",
        )
    else:
        meters = MetricLogger(delimiter="  ")

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    evaluate_period = cfg.SOLVER.EVALUATE_PERIOD
    arguments["max_epoch"] = cfg.SOLVER.NUM_EPOCHS
    arguments["distributed"] = distributed

    do_train(
        model,
        data_loader,
        data_loader_val,
        optimizer,
        scheduler,
        checkpointer,
        meters,
        device,
        checkpoint_period,
        evaluate_period,
        arguments,
    )


def main():
    set_random_seed()

    parser = argparse.ArgumentParser(description="PyTorch Person Search Training")
    parser.add_argument(
        "--root",
        default="./",
        help="root path",
        type=str,
    )
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--resume-from",
        help="the checkpoint file to resume from",
        type=str,
    )
    parser.add_argument(
        "--local_rank",
        default=0,
        type=int,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--use-tensorboard",
        dest="use_tensorboard",
        help="Use tensorboardX logger (Requires tensorboardX and tensorflow installed)",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.ROOT = args.root
    cfg.freeze()

    output_dir = os.path.join(
        args.root, "./output", "/".join(args.config_file.split("/")[-2:])[:-5]
    )
    makedir(output_dir)

    logger = setup_logger("PersonSearch", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    train(
        cfg,
        output_dir,
        args.local_rank,
        args.distributed,
        args.resume_from,
        args.use_tensorboard,
    )


if __name__ == "__main__":
    main()
