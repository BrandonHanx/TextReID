import copy
import logging
import math

import timm.models.helpers as helpers
import timm.models.swin_transformer as timmswin
import torch
import torch.nn.functional as F


class SwinTransformer(timmswin.SwinTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = self.num_features
        del self.head

    def forward(self, x):
        x = self.forward_features(x)
        return x


def resize_pos_embed(posemb, posemb_new, gs_new):
    # Rescale the grid of position embeddings when loading from state_dict.
    logger = logging.getLogger("PersonSearch.train")
    logger.info(
        "Resized position embedding: {} to {}".format(posemb.shape, posemb_new.shape)
    )
    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
    gs_old = int(math.sqrt(len(posemb_grid)))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(
        posemb_grid, size=gs_new, mode="bilinear", align_corners=False
    )
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if "model" in state_dict:
        # For deit models
        state_dict = state_dict["model"]
    for k, v in state_dict.items():
        if "patch_embed.proj.weight" in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == "pos_embed" and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            gs_new = (
                model.patch_embed.img_size[0] // model.patch_embed.patch_size[0],
                model.patch_embed.img_size[1] // model.patch_embed.patch_size[1],
            )
            v = resize_pos_embed(v, model.pos_embed, gs_new)
        elif k.split(".")[-1] in ["attn_mask", "relative_position_index"]:
            continue
        elif k in ["head.weight", "head.bias"]:
            continue
        out_dict[k] = v
    return out_dict


def _create_swin_transformer(variant, pretrained=False, **kwargs):
    default_cfg = copy.deepcopy(timmswin.default_cfgs[variant])
    helpers.overlay_external_default_cfg(default_cfg, kwargs)
    default_num_classes = default_cfg["num_classes"]
    default_img_size = default_cfg["input_size"][-2:]

    num_classes = kwargs.pop("num_classes", default_num_classes)
    img_size = kwargs.pop("img_size", default_img_size)
    if kwargs.get("features_only", None):
        raise RuntimeError(
            "features_only not implemented for Vision Transformer models."
        )

    model = helpers.build_model_with_cfg(
        SwinTransformer,
        variant,
        pretrained,
        default_cfg=default_cfg,
        img_size=img_size,
        num_classes=num_classes,
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs
    )

    return model


def swin_small_patch4_window7_224(cfg):
    """Swin-S @ 224x224, trained ImageNet-1k"""
    model_kwargs = dict(
        img_size=(cfg.INPUT.HEIGHT, cfg.INPUT.WIDTH),
        patch_size=4,
        window_size=7,
        embed_dim=96,
        depths=(2, 2, 18, 2),
        num_heads=(3, 6, 12, 24),
    )
    return _create_swin_transformer(
        "swin_small_patch4_window7_224", pretrained=True, **model_kwargs
    )


def swin_tiny_patch4_window7_224(cfg):
    """Swin-T @ 224x224, trained ImageNet-1k"""
    model_kwargs = dict(
        img_size=(cfg.INPUT.HEIGHT, cfg.INPUT.WIDTH),
        patch_size=4,
        window_size=7,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
    )
    return _create_swin_transformer(
        "swin_tiny_patch4_window7_224", pretrained=True, **model_kwargs
    )


def build_swin(cfg):
    return swin_small_patch4_window7_224(cfg)
