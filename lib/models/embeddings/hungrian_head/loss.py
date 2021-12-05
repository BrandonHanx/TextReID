import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import lib.models.losses as losses


def hungrian_loss(local_similarity, scale=10, margin=0.7):
    loss = torch.log(1 + torch.exp(scale * (margin - local_similarity)))
    return loss.sum()


class LossComputation(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.global_projection = Parameter(
            torch.randn(cfg.MODEL.EMBEDDING.FEATURE_SIZE, cfg.MODEL.NUM_CLASSES),
            requires_grad=True,
        )
        nn.init.xavier_uniform_(self.global_projection.data, gain=1)

    def forward(
        self,
        visual_embed,
        textual_embed,
        local_similarity,
        captions,
    ):
        labels = torch.stack([caption.get_field("id") for caption in captions]).long()
        loss = {
            "instance_loss": losses.instance_loss(
                self.global_projection, visual_embed, textual_embed, labels
            ),
            "global_align_loss": losses.global_align_loss(
                visual_embed, textual_embed, labels
            ),
            "local_align_loss": hungrian_loss(local_similarity),
        }
        return loss


def make_loss_evaluator(cfg):
    return LossComputation(cfg)
