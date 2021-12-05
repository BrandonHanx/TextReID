import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import lib.models.losses as losses


def kl_div(pred, label):
    return F.kl_div(
        F.log_softmax(pred, -1), F.softmax(label, -1), reduction="batchmean"
    )


def soft_hinge(score, alpha=0.6, beta=0.4, scale_pos=1, scale_neg=1):
    half_num = int(score.shape[0] / 2)
    loss_pos = torch.log(1 + torch.exp(-scale_pos * (score[:half_num] - alpha)))
    loss_neg = torch.log(1 + torch.exp(scale_neg * (score[half_num:] - beta)))
    loss = (loss_pos.sum() + loss_neg.sum()) / half_num
    return loss


def triplet(score, margin=0.3):
    half_num = int(score.shape[0] / 2)
    pos = score[:half_num]
    neg = score[half_num:]
    loss = torch.clamp(margin + neg - pos, 0).mean()
    return loss


def bce(score):
    half_num = int(score.shape[0] / 2)
    label = torch.cat([torch.ones(half_num), torch.zeros(half_num)]).cuda().long()
    loss = F.cross_entropy(score, label)
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

    @staticmethod
    def forward_itm(score, lamda=10.0):
        loss = {"itm_loss": lamda * bce(score)}
        return loss

    @staticmethod
    def forward_mpm(predicted_patch, masked_patch, lamda=10.0):
        loss = {"mpm_loss": lamda * kl_div(predicted_patch, masked_patch)}
        return loss

    @staticmethod
    def forward_mwm(predicted_word, masked_word, lamda=10.0):
        loss = {"mwm_loss": lamda * kl_div(predicted_word, masked_word)}
        return loss

    def forward_cmr(
        self,
        visual_embed,
        textual_embed,
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
            "global_align_loss": losses.global_align_loss(
                visual_embed, textual_embed, labels
            ),
        }
        return loss


def make_loss_evaluator(cfg):
    return LossComputation(cfg)
