import torch.nn as nn
import torch.nn.functional as F

from .loss import make_loss_evaluator


class SimpleHead(nn.Module):
    def __init__(
        self,
        cfg,
        visual_size,
        textual_size,
    ):
        super().__init__()
        self.embed_size = cfg.MODEL.EMBEDDING.FEATURE_SIZE
        self.bnneck = cfg.MODEL.EMBEDDING.BNNECK

        self.visual_embed_layer = nn.Linear(visual_size, self.embed_size)
        self.textual_embed_layer = nn.Linear(textual_size, self.embed_size)

        if cfg.MODEL.VISUAL_MODEL.split("_")[0] == "vit":
            self.avgpool = None
        elif cfg.MODEL.VISUAL_MODEL in [
            "hbvit",
            "pit",
            "pvt",
            "swin",
            "m_resnet",
            "m_resnet50",
            "m_resnet101",
            "clip_resnet",
        ]:
            self.avgpool = None
        else:
            if cfg.MODEL.RESNET.ATTN_POOL:
                self.avgpool = None
            else:
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.shared_embed_layer = None
        if cfg.MODEL.EMBEDDING.SHARED_LAYER:
            self.shared_embed_layer = nn.Sequential(
                nn.Linear(self.embed_size, self.embed_size),
                nn.ReLU(),
                nn.Linear(self.embed_size, self.embed_size),
            )

        if self.bnneck:
            self.visual_bnneck = nn.BatchNorm1d(self.embed_size)
            self.textual_bnneck = nn.BatchNorm1d(self.embed_size)
            self.visual_bnneck.bias.requires_grad_(False)  # no shift
            self.textual_bnneck.bias.requires_grad_(False)  # no shift

        self.loss_evaluator = make_loss_evaluator(cfg)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, visual_feature, textual_feature, captions):
        batch_size = visual_feature.size(0)

        if self.avgpool is not None:
            visual_feature = self.avgpool(visual_feature)

        visual_embed = visual_feature.view(batch_size, -1)
        textual_embed = textual_feature.view(batch_size, -1)

        visual_embed = self.visual_embed_layer(visual_embed)
        textual_embed = self.textual_embed_layer(textual_embed)

        if self.shared_embed_layer is not None:
            visual_embed = self.shared_embed_layer(
                F.normalize(visual_embed, p=2, dim=1)
            )
            textual_embed = self.shared_embed_layer(
                F.normalize(textual_embed, p=2, dim=1)
            )

        if self.bnneck:
            visual_embed_bn = self.visual_bnneck(visual_embed)
            textual_embed_bn = self.textual_bnneck(textual_embed)

            if self.training:
                losses = self.loss_evaluator(
                    visual_embed,
                    textual_embed,
                    captions,
                    visual_embed_bn,
                    textual_embed_bn,
                )
                return None, losses

            outputs = list()
            outputs.append(visual_embed_bn)
            outputs.append(textual_embed_bn)
            return outputs, None

        if self.training:
            losses = self.loss_evaluator(visual_embed, textual_embed, captions)
            return None, losses

        outputs = list()
        outputs.append(visual_embed)
        outputs.append(textual_embed)
        return outputs, None


def build_simple_head(cfg, visual_size, textual_size):
    model = SimpleHead(cfg, visual_size, textual_size)
    return model
