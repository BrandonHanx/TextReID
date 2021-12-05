import copy
import logging
import math
from functools import partial

import numpy as np
import timm.models.vision_transformer as ViTcls
import torch
import torch.nn.functional as F
from timm.models.helpers import load_checkpoint, load_pretrained


class ViT(ViTcls.VisionTransformer):
    def __init__(self, mode, **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self.out_channels = self.embed_dim

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        if self.mode == "jpm":
            for blk in self.blocks[:-1]:
                x = blk(x)
            return x

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        if self.mode == "first":
            return x[:, 0]
        if self.mode == "average":
            return x[:, 1:].mean(dim=1)
        return NotImplementedError


class ViTWithJPM(torch.nn.Module):
    def __init__(self, vit, shift_offset=5, shuffle_group=4):
        super().__init__()
        self.vit = vit
        self.jpm = copy.deepcopy(
            self.vit.blocks[-1]
        )  # initialize the weight same as last layer
        self.jpm_norm = copy.deepcopy(self.vit.norm)
        self.shift_offset = shift_offset
        self.shuffle_group = shuffle_group

    def forward(self, x):
        x = self.vit(x)
        global_feat = self.vit.blocks[-1](x)
        global_feat = self.vit.norm(global_feat)[:, 0]

        cls_token = x[:, 0].unsqueeze(dim=1)
        feat_len = x.shape[1] - 1

        local_feat = torch.cat(
            [x[:, self.shift_offset + 1 :], x[:, 1 : self.shift_offset + 1]], dim=1
        )  # shift
        random_idx = list(np.random.permutation(feat_len))
        local_feat = local_feat[:, random_idx]  # shuffle

        jpm_feats = [global_feat]
        group_idxs = np.linspace(0, feat_len, self.shuffle_group + 1, dtype=int)
        for i in range(len(group_idxs) - 1):
            feat = torch.cat(
                [cls_token, local_feat[:, group_idxs[i] : group_idxs[i + 1]]], dim=1
            )
            feat = self.jpm(feat)
            feat = self.jpm_norm(feat)
            jpm_feats.append(feat[:, 0])

        return jpm_feats


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


def checkpoint_filter_fn(state_dict, model, gs_new):
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
            v = resize_pos_embed(v, model.pos_embed, gs_new)
        out_dict[k] = v
    return out_dict


def _create_vit(variant, mode, img_size, ckpt, patch_size, **kwargs):
    model = ViT(mode=mode, img_size=img_size, **kwargs)
    model.default_cfg = ViTcls.default_cfgs[variant]
    logger = logging.getLogger("PersonSearch.train")

    if ckpt:
        logger.info("Load pretrained ViT from {}".format(ckpt))
        load_checkpoint(model, ckpt)
    else:
        logger.info("Load pretrained ViT from timm base")
        gs_new = (int(img_size[0] / patch_size), int(img_size[1] / patch_size))
        load_pretrained(
            model, filter_fn=partial(checkpoint_filter_fn, model=model, gs_new=gs_new)
        )
    return model


model_archs = {}
model_archs["vit_deit_small_patch16_224"] = dict(
    patch_size=16, embed_dim=384, depth=12, num_heads=6
)
model_archs["vit_base_patch16_224_in21k"] = dict(
    patch_size=16, embed_dim=768, depth=12, num_heads=12
)
model_archs["jpm_deit_small_patch16_224"] = dict(
    patch_size=16, embed_dim=384, depth=12, num_heads=6
)
model_archs["vit_deit_base_patch16_224"] = dict(
    patch_size=16, embed_dim=768, depth=12, num_heads=12
)


def build_vit(cfg):
    arch = cfg.MODEL.VISUAL_MODEL
    model_arch = model_archs[arch]
    if arch == "jpm_deit_small_patch16_224":
        vit = _create_vit(
            variant="vit_deit_small_patch16_224",
            mode="jpm",
            img_size=(cfg.INPUT.HEIGHT, cfg.INPUT.WIDTH),
            ckpt=cfg.MODEL.VIT.CKPT,
            **model_arch
        )
        model = ViTWithJPM(vit)
    else:
        model = _create_vit(
            variant=cfg.MODEL.VISUAL_MODEL,
            mode=cfg.MODEL.VIT.MODE,
            img_size=(cfg.INPUT.HEIGHT, cfg.INPUT.WIDTH),
            ckpt=cfg.MODEL.VIT.CKPT,
            **model_arch
        )
    if cfg.MODEL.FREEZE:
        model.pos_embed.requires_grad = False
        for m in [model.patch_embed, model.blocks[:-1]]:
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
    return model
