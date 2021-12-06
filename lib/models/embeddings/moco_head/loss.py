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
        self.epsilon = cfg.MODEL.EMBEDDING.EPSILON
        # self.T = Parameter(torch.tensor(0.07), requires_grad=True)
        self.T = 0.07
        nn.init.xavier_uniform_(self.projection.data, gain=1)

    def forward(self, v_embed, t_embed, v_pos, v_neg, t_pos, t_neg, labels):
        loss = {
            "instance_loss": losses.instance_loss(
                self.projection,
                v_embed,
                t_embed,
                labels,
                epsilon=self.epsilon,
            ),
            "infonce_loss": losses.infonce_loss(
                v_pos,
                v_neg,
                t_pos,
                t_neg,
                self.T,
            ),
            "global_align_loss": losses.global_align_loss(v_embed, t_embed, labels),
        }
        return loss


def make_loss_evaluator(cfg):
    return LossComputation(cfg)
