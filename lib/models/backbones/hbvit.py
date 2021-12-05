import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

from .ibn_a import resnet50_ibn_a
from .resnet import ResNet, model_archs


class HybridViT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        backbone = cfg.MODEL.HBVIT.BACKBONE
        if backbone == "resnet50":
            self.backbone = ResNet(
                model_archs["resnet50"],
                res5_stride=cfg.MODEL.RESNET.RES5_STRIDE,
                res5_dilation=cfg.MODEL.RESNET.RES5_DILATION,
                mode=cfg.MODEL.EMBEDDING.EMBED_HEAD,
                pretrained=True,
            )
        elif backbone == "resnet50_ibn_a":
            self.backbone = resnet50_ibn_a(
                cfg.MODEL.RESNET.RES5_STRIDE, pretrained=True
            )
        down_ratio = 32
        if cfg.MODEL.RESNET.RES5_STRIDE == 1:
            down_ratio = 16

        embed_dim = cfg.MODEL.HBVIT.EMBED_DIM
        patch_size = cfg.MODEL.HBVIT.PATCH_SIZE
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        num_patches = (cfg.INPUT.HEIGHT // down_ratio // patch_size) * (
            cfg.INPUT.WIDTH // down_ratio // patch_size
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.trans_layers = nn.ModuleList(
            [
                Block(dim=embed_dim, num_heads=cfg.MODEL.HBVIT.NUM_HEADS)
                for i in range(cfg.MODEL.HBVIT.DEPTH)
            ]
        )
        self.proj_conv = nn.Conv2d(
            self.backbone.out_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.out_channels = embed_dim
        self.whole = cfg.MODEL.WHOLE

    def forward(self, x):
        B = x.shape[0]
        x = self.backbone(x)
        x = self.proj_conv(x).flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        for blk in self.trans_layers:
            x = blk(x)

        x = self.norm(x)
        if self.whole:
            return x
        return x[:, 0]


def build_hybrid_vit(cfg):
    return HybridViT(cfg)
