import torch

from .lr_scheduler import LRSchedulerWithWarmup


def make_optimizer(cfg, model):
    params = []

    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if cfg.SOLVER.OPTIMIZER == "SGD":
        optimizer = torch.optim.SGD(
            params, lr=cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.SGD_MOMENTUM
        )
    elif cfg.SOLVER.OPTIMIZER == "Adam":
        optimizer = torch.optim.Adam(
            params,
            lr=cfg.SOLVER.BASE_LR,
            betas=(cfg.SOLVER.ADAM_ALPHA, cfg.SOLVER.ADAM_BETA),
            eps=1e-8,
        )
    elif cfg.SOLVER.OPTIMIZER == "AdamW":
        optimizer = torch.optim.AdamW(
            params,
            lr=cfg.SOLVER.BASE_LR,
            betas=(cfg.SOLVER.ADAM_ALPHA, cfg.SOLVER.ADAM_BETA),
            eps=1e-8,
        )
    else:
        NotImplementedError

    return optimizer


def make_lr_scheduler(cfg, optimizer):
    return LRSchedulerWithWarmup(
        optimizer,
        milestones=cfg.SOLVER.STEPS,
        gamma=cfg.SOLVER.GAMMA,
        warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
        warmup_epochs=cfg.SOLVER.WARMUP_EPOCHS,
        warmup_method=cfg.SOLVER.WARMUP_METHOD,
        total_epochs=cfg.SOLVER.NUM_EPOCHS,
        mode=cfg.SOLVER.LRSCHEDULER,
        target_lr=cfg.SOLVER.TARGET_LR,
        power=cfg.SOLVER.POWER,
    )
