from torch import nn

from .backbones import build_textual_model, build_visual_model
from .embeddings import build_embed
from .embeddings.moco_head.head import build_moco_head


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.visual_model = build_visual_model(cfg)
        self.textual_model = build_textual_model(cfg)

        if cfg.MODEL.EMBEDDING.EMBED_HEAD == "moco":
            self.embed_model = build_moco_head(
                cfg, self.visual_model, self.textual_model
            )
            self.embed_type = "moco"
        else:
            self.embed_model = build_embed(
                cfg, self.visual_model.out_channels, self.textual_model.out_channels
            )
            self.embed_type = "normal"

    def forward(self, images, captions):
        if self.embed_type == "moco":
            return self.embed_model(images, captions)

        visual_feat = self.visual_model(images)
        textual_feat = self.textual_model(captions)

        outputs_embed, losses_embed = self.embed_model(
            visual_feat, textual_feat, captions
        )

        if self.training:
            losses = {}
            losses.update(losses_embed)
            return losses

        return outputs_embed


def build_model(cfg):
    return Model(cfg)
