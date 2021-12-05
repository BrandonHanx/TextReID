import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import lib.models.losses as losses


class LossComputation(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.projection = Parameter(
            torch.randn(cfg.MODEL.EMBEDDING.FEATURE_SIZE, cfg.MODEL.NUM_CLASSES),
            requires_grad=True,
        )
        nn.init.xavier_uniform_(self.projection.data, gain=1)

    def forward(
        self,
        visual_embed,
        textual_embed,
        captions,
        visual_domain_logits,
        textual_domain_logits,
    ):
        labels = torch.stack([caption.get_field("id") for caption in captions]).long()

        loss = {
            "instance_loss": losses.instance_loss(
                self.projection, visual_embed, textual_embed, labels
            ),
            "global_align_loss": losses.global_align_loss(
                visual_embed, textual_embed, labels
            ),
            "domain_loss": -0.0005
            * losses.domain_loss(visual_domain_logits, textual_domain_logits),
        }
        return loss


def make_loss_evaluator(cfg):
    return LossComputation(cfg)
