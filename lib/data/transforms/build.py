from . import transforms as T


def build_transforms(cfg, is_train=True):
    height = cfg.INPUT.HEIGHT
    width = cfg.INPUT.WIDTH
    ratio = cfg.INPUT.DOWNSAMPLE_RATIO
    use_aug = cfg.INPUT.USE_AUG

    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
    )

    if is_train:
        if use_aug:
            transform = T.Compose(
                [
                    T.Resize((height, width), ratio=1.0),
                    T.RandomHorizontalFlip(0.5),
                    T.Pad(cfg.INPUT.PADDING),
                    T.RandomCrop((height, width)),
                    T.Resize((height, width), ratio=ratio),
                    T.ToTensor(),
                    normalize_transform,
                    T.RandomErasing(mean=cfg.INPUT.PIXEL_MEAN),
                ]
            )
        else:
            transform = T.Compose(
                [
                    T.Resize((height, width), ratio=ratio),
                    T.RandomHorizontalFlip(0.5),
                    T.ToTensor(),
                    normalize_transform,
                ]
            )
    else:
        transform = T.Compose(
            [
                T.Resize((height, width), ratio=ratio),
                T.ToTensor(),
                normalize_transform,
            ]
        )
    return transform


def build_crop_transforms(cfg):
    transform = T.Split(cfg.MODEL.NUM_PARTS, cfg.DATASETS.BIN_SEG)
    return transform
