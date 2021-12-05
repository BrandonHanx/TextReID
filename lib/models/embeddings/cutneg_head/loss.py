import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import lib.models.losses as losses


def aug_align_loss(visual_embed, neg_textual_embed, scale=40, beta=0.4):
    N, D = visual_embed.shape
    visual_embed = visual_embed.unsqueeze(1)  # b x 1 x d
    neg_textual_embed = neg_textual_embed.reshape(-1, N, D)  # ? x b x d
    visual_embed = F.normalize(visual_embed, dim=-1)
    neg_textual_embed = F.normalize(neg_textual_embed, dim=-1)
    similarity = torch.bmm(visual_embed, neg_textual_embed.permute(1, 2, 0)).view(-1)
    loss = torch.log(1 + torch.exp(scale * (similarity - beta))).sum() / N
    return loss


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
        neg_textual_embed,
        captions,
    ):
        labels = torch.stack([caption.get_field("id") for caption in captions]).long()
        loss = {
            "instance_loss": losses.instance_loss(
                self.projection,
                visual_embed,
                textual_embed,
                labels,
                epsilon=self.epsilon,
            ),
            "global_align_loss": losses.global_align_loss(
                visual_embed,
                textual_embed,
                labels,
            ),
            "aug_align_loss": aug_align_loss(visual_embed, neg_textual_embed),
        }
        return loss


def make_loss_evaluator(cfg):
    return LossComputation(cfg)
