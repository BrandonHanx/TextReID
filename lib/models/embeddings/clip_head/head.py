import torch.nn as nn

from .loss import make_loss_evaluator


class CLIPHead(nn.Module):
    def __init__(
        self,
        cfg,
        visual_size,
        textual_size,
    ):
        super().__init__()
        #         self.embed_size = cfg.MODEL.EMBEDDING.FEATURE_SIZE
        #         self.visual_embed_layer = nn.Linear(visual_size, self.embed_size)
        #         self.textual_embed_layer = nn.Linear(textual_size, self.embed_size)

        self.loss_evaluator = make_loss_evaluator(cfg)

    def forward(self, visual_embed, textual_embed, captions):
        #         visual_embed = self.visual_embed_layer(visual_embed)
        #         textual_embed = self.textual_embed_layer(textual_embed)

        if self.training:
            losses = self.loss_evaluator(visual_embed, textual_embed, captions)
            return None, losses

        outputs = list()
        outputs.append(visual_embed)
        outputs.append(textual_embed)
        return outputs, None


def build_clip_head(cfg, visual_size, textual_size):
    model = CLIPHead(cfg, visual_size, textual_size)
    return model
