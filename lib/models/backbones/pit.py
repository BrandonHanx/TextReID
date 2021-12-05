import copy
import logging
import re

import timm.models.helpers as helpers
import timm.models.pit as timmpit
import torch.nn.functional as F


class PiT(timmpit.PoolingVisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = self.embed_dim

    def forward(self, x):
        x = self.forward_features(x)
        x_cls = x[:, 0]
        if self.num_tokens > 1:
            x_dist = x[:, 1]
            return (x_cls + x_dist) / 2
        return x_cls


def resize_pos_embed(posemb, posemb_new):
    # Rescale the grid of position embeddings when loading from state_dict.
    logger = logging.getLogger("PersonSearch.train")
    logger.info(
        "Resized position embedding: {} to {}".format(posemb.shape, posemb_new.shape)
    )
    posemb = F.interpolate(
        posemb, size=posemb_new.shape[-2:], mode="bilinear", align_corners=False
    )
    return posemb


def checkpoint_filter_fn(state_dict, model):
    out_dict = {}
    p_blocks = re.compile(r"pools\.(\d)\.")
    for k, v in state_dict.items():
        if k == "pos_embed" and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(v, model.pos_embed)
        k = p_blocks.sub(lambda exp: f"transformers.{int(exp.group(1))}.pool.", k)
        out_dict[k] = v
    return out_dict


def _create_pit(variant, pretrained=False, **kwargs):
    default_cfg = copy.deepcopy(timmpit.default_cfgs[variant])
    helpers.overlay_external_default_cfg(default_cfg, kwargs)
    default_num_classes = default_cfg["num_classes"]
    default_img_size = default_cfg["input_size"][-2:]
    img_size = kwargs.pop("img_size", default_img_size)
    num_classes = kwargs.pop("num_classes", default_num_classes)

    if kwargs.get("features_only", None):
        raise RuntimeError(
            "features_only not implemented for Vision Transformer models."
        )

    model = helpers.build_model_with_cfg(
        PiT,
        variant,
        pretrained,
        default_cfg=default_cfg,
        img_size=img_size,
        num_classes=num_classes,
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs,
    )

    return model


def pit_s_distilled_224(cfg):
    model_kwargs = dict(
        img_size=(cfg.INPUT.HEIGHT, cfg.INPUT.WIDTH),
        patch_size=16,
        stride=8,
        base_dims=[48, 48, 48],
        depth=[2, 6, 4],
        heads=[3, 6, 12],
        mlp_ratio=4,
        distilled=True,
    )
    return _create_pit("pit_s_distilled_224", pretrained=True, **model_kwargs)


def pit_b_distilled_224(cfg):
    model_kwargs = dict(
        img_size=(cfg.INPUT.HEIGHT, cfg.INPUT.WIDTH),
        patch_size=14,
        stride=7,
        base_dims=[64, 64, 64],
        depth=[3, 6, 4],
        heads=[4, 8, 16],
        mlp_ratio=4,
        distilled=True,
    )
    return _create_pit("pit_b_distilled_224", pretrained=True, **model_kwargs)


def build_pit(cfg):
    return pit_b_distilled_224(cfg)
