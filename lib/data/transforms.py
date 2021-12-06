import torchvision.transforms as T


def build_transforms(cfg, is_train=True):
    height = cfg.INPUT.HEIGHT
    width = cfg.INPUT.WIDTH
    use_aug = cfg.INPUT.USE_AUG

    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
    )

    if is_train:
        if use_aug:
            transform = T.Compose(
                [
                    T.Resize((height, width)),
                    T.RandomHorizontalFlip(0.5),
                    T.Pad(cfg.INPUT.PADDING),
                    T.RandomCrop((height, width)),
                    T.ToTensor(),
                    normalize_transform,
                    T.RandomErasing(scale=(0.02, 0.4), value=cfg.INPUT.PIXEL_MEAN),
                ]
            )
        else:
            transform = T.Compose(
                [
                    T.Resize((height, width)),
                    T.RandomHorizontalFlip(0.5),
                    T.ToTensor(),
                    normalize_transform,
                ]
            )
    else:
        transform = T.Compose(
            [
                T.Resize((height, width)),
                T.ToTensor(),
                normalize_transform,
            ]
        )
    return transform
