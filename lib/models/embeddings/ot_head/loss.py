import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import lib.models.losses as losses


def ot_loss(ot_dist, margin=0.2):
    half_num = int(ot_dist.shape[0] / 2)
    ot_dist = 1 - ot_dist
    ot_pos = ot_dist[:half_num]
    ot_neg = ot_dist[half_num:]
    ot_loss = torch.clamp(ot_neg + margin - ot_pos, 0).sum()
    return ot_loss


class LossComputation(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.epsilon = cfg.MODEL.EMBEDDING.EPSILON

        self.projection = Parameter(
            torch.randn(cfg.MODEL.EMBEDDING.FEATURE_SIZE, cfg.MODEL.NUM_CLASSES),
            requires_grad=True,
        )
        nn.init.xavier_uniform_(self.projection.data, gain=1)

    def forward(
        self,
        visual_embed,
        textual_embed,
        dist_mat,
        labels,
    ):
        loss = {
            "instance_loss": losses.instance_loss(
                self.projection,
                visual_embed,
                textual_embed,
                labels,
                epsilon=self.epsilon,
            ),
            "ot_align_loss": losses.global_align_loss_from_sim(dist_mat, labels),
        }
        return loss


def make_loss_evaluator(cfg):
    return LossComputation(cfg)
