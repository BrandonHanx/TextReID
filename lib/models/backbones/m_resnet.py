import logging
import math
import random
import re
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn

from .ibn_a import IBN


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, ibna=False):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        if ibna:
            self.bn1 = IBN(planes)
        else:
            self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(
                OrderedDict(
                    [
                        ("-1", nn.AvgPool2d(stride)),
                        (
                            "0",
                            nn.Conv2d(
                                inplanes,
                                planes * self.expansion,
                                1,
                                stride=1,
                                bias=False,
                            ),
                        ),
                        ("1", nn.BatchNorm2d(planes * self.expansion)),
                    ]
                )
            )

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(
        self,
        spacial_dim,
        embed_dim,
        num_heads,
        output_dim=None,
        patch_size=1,
        whole=False,
    ):
        super().__init__()
        self.spacial_dim = spacial_dim
        self.proj_conv = None
        if patch_size > 1:
            self.proj_conv = nn.Conv2d(
                embed_dim,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
                bias=False,
            )
        self.positional_embedding = nn.Parameter(
            torch.randn(
                (spacial_dim[0] // patch_size) * (spacial_dim[1] // patch_size) + 1,
                embed_dim,
            )
            / embed_dim ** 0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
        self.whole = whole

    def forward(self, x):
        if self.proj_conv is not None:
            x = self.proj_conv(x)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(
            2, 0, 1
        )  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x,
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
            ),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )

        if self.whole:
            return x.transpose(0, 1)
        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(
        self,
        layers,
        output_dim,
        heads,
        last_stride=1,
        input_resolution=(224, 224),
        width=64,
        whole=False,
        ibna=False,
        patch_mix=False,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.out_channels = output_dim
        self.input_resolution = input_resolution
        self.whole = whole
        self.patch_mix = patch_mix

        # the 3-layer stem
        self.conv1 = nn.Conv2d(
            3, width // 2, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(
            width // 2, width // 2, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0], ibna=ibna)
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2, ibna=ibna)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2, ibna=ibna)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=last_stride)

        embed_dim = width * 32  # the ResNet feature dimension
        down_ratio = 16 if last_stride == 1 else 32
        spacial_dim = (
            input_resolution[0] // down_ratio,
            input_resolution[1] // down_ratio,
        )
        self.attnpool = AttentionPool2d(
            spacial_dim, embed_dim, heads, output_dim, whole=whole
        )

    def _make_layer(self, planes, blocks, stride=1, ibna=False):
        layers = [Bottleneck(self._inplanes, planes, stride, ibna=ibna)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes, ibna=ibna))

        return nn.Sequential(*layers)

    @staticmethod
    def _patch_mix(patches, k=4, num_shuffle_patch=48, p=0.5):
        if random.random() > p:
            return patches
        N, C, H, W = patches.shape
        origin_idx = torch.arange(N).reshape(-1, k)  # b/4 x 4
        shuffle_perm = torch.randperm(k)
        shuffle_idx = origin_idx[:, shuffle_perm].view(-1)  # b

        patches = patches.reshape(N, C, H * W)
        idx = random.randint(0, H * W - num_shuffle_patch)
        patches = torch.cat(
            (
                patches[:, :, :idx],
                patches[shuffle_idx, :, idx : idx + num_shuffle_patch],
                patches[:, :, idx + num_shuffle_patch :],
            ),
            dim=-1,
        )
        patches = patches.reshape(N, C, H, W)
        return patches

    def forward(self, x):
        def stem(x):
            for conv, bn in [
                (self.conv1, self.bn1),
                (self.conv2, self.bn2),
                (self.conv3, self.bn3),
            ]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.patch_mix and self.training:
            x = self._patch_mix(x)
        x = self.attnpool(x)

        if self.whole:
            return x[:, 0], x[:, 1:]
        return x


def resize_pos_embed(posemb, gs_new):
    # Rescale the grid of position embeddings when loading from state_dict.
    logger = logging.getLogger("PersonSearch.train")
    posemb_tok, posemb_grid = posemb[:1], posemb[1:]
    gs_old = int(math.sqrt(len(posemb_grid)))
    logger.info("Resized position embedding: {} to {}".format((gs_old, gs_old), gs_new))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(
        posemb_grid, size=gs_new, mode="bilinear", align_corners=False
    )
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=0)
    return posemb


def state_filter(state_dict, final_stage_resolution, ibna):
    logger = logging.getLogger("PersonSearch.train")
    out_dict = {}
    for k, v in state_dict.items():
        if k.startswith("visual."):
            k = k[7:]
        if k == "attnpool.positional_embedding" and final_stage_resolution != (7, 7):
            v = resize_pos_embed(v, final_stage_resolution)
        if ibna and re.match(r"layer[123]\..*\.bn1.*", k):
            bn_k = k.split(".")
            if bn_k[-1] == "num_batches_tracked":
                continue
            dim = v.shape[0]
            bn_k.insert(-1, "BN")
            bn_k = ".".join(bn_k)
            logger.info("Change {} to {}".format(k, bn_k))
            out_dict[bn_k] = v[: int(dim / 2)]

            in_k = k.split(".")
            if in_k[-1] in ["running_mean", "running_var"]:
                continue
            in_k.insert(-1, "IN")
            in_k = ".".join(in_k)
            logger.info("Change {} to {}".format(k, in_k))
            out_dict[in_k] = v[int(dim / 2) :]
        else:
            out_dict[k] = v
    return out_dict


def modified_resnet50(
    input_resolution,
    last_stride,
    whole=False,
    ibna=False,
    patch_mix=False,
    pretrained=False,
):
    model = ModifiedResNet(
        layers=[3, 4, 6, 3],
        output_dim=1024,
        heads=32,
        last_stride=last_stride,
        input_resolution=input_resolution,
        whole=whole,
        ibna=ibna,
        patch_mix=patch_mix,
    )
    if pretrained:
        p = torch.jit.load("pretrained/clip/RN50.pt").state_dict()
        model.load_state_dict(
            state_filter(
                p,
                final_stage_resolution=model.attnpool.spacial_dim,
                ibna=ibna,
            ),
            strict=False,
        )
    return model


def modified_resnet101(
    input_resolution,
    last_stride,
    whole=False,
    ibna=False,
    patch_mix=False,
    pretrained=False,
):
    model = ModifiedResNet(
        layers=[3, 4, 23, 3],
        output_dim=512,
        heads=32,
        last_stride=last_stride,
        input_resolution=input_resolution,
        whole=whole,
        ibna=ibna,
        patch_mix=patch_mix,
    )
    if pretrained:
        p = torch.jit.load("pretrained/clip/RN101.pt").state_dict()
        model.load_state_dict(
            state_filter(
                p,
                final_stage_resolution=model.attnpool.spacial_dim,
                ibna=ibna,
            ),
            strict=False,
        )
    return model


def build_m_resnet(cfg):
    if cfg.MODEL.VISUAL_MODEL in ["m_resnet50", "m_resnet"]:
        model = modified_resnet50(
            (cfg.INPUT.HEIGHT, cfg.INPUT.WIDTH),
            cfg.MODEL.RESNET.RES5_STRIDE,
            cfg.MODEL.WHOLE,
            cfg.MODEL.RESNET.IBNA,
            cfg.MODEL.RESNET.PATCH_MIX,
            pretrained=True,
        )
    elif cfg.MODEL.VISUAL_MODEL == "m_resnet101":
        model = modified_resnet101(
            (cfg.INPUT.HEIGHT, cfg.INPUT.WIDTH),
            cfg.MODEL.RESNET.RES5_STRIDE,
            cfg.MODEL.WHOLE,
            cfg.MODEL.RESNET.IBNA,
            cfg.MODEL.RESNET.PATCH_MIX,
            pretrained=True,
        )
    return model
