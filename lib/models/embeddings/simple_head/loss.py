import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import lib.models.losses as losses


class LossComputation(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cmpc = cfg.MODEL.EMBEDDING.CMPC
        self.cmpm = cfg.MODEL.EMBEDDING.CMPM
        self.mixture = cfg.MODEL.EMBEDDING.MIXTURE
        self.bnneck = cfg.MODEL.EMBEDDING.BNNECK
        self.epsilon = cfg.MODEL.EMBEDDING.EPSILON
        self.learn_scale = cfg.MODEL.EMBEDDING.LEARN_SCALE

        if self.learn_scale:
            self.scale_pos = Parameter(torch.tensor(10.0), requires_grad=True)
            self.scale_neg = Parameter(torch.tensor(40.0), requires_grad=True)
        else:
            self.scale_pos = 10.0
            self.scale_neg = 40.0

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
        visual_embed_bn=None,
        textual_embed_bn=None,
    ):
        labels = torch.stack([caption.get_field("id") for caption in captions]).long()
        if self.cmpm and self.cmpc:
            loss = {
                "cmpc_loss": losses.cmpc_loss(
                    self.projection, visual_embed, textual_embed, labels
                ),
                "cmpm_loss": losses.cmpm_loss(visual_embed, textual_embed, labels),
            }
            return loss
        if not self.cmpc and self.bnneck:
            loss = {
                "instance_loss": losses.instance_loss(
                    self.projection, visual_embed_bn, textual_embed_bn, labels
                ),
                "global_align_loss": losses.global_align_loss(
                    visual_embed, textual_embed, labels, self.mixture
                ),
            }
            return loss
        if self.cmpc and not self.bnneck:  # baseline
            loss = {
                "cmpc_loss": losses.cmpc_loss(
                    self.projection, visual_embed, textual_embed, labels
                ),
                "global_align_loss": losses.global_align_loss(
                    visual_embed, textual_embed, labels, self.mixture
                ),
            }
            return loss
        if self.cmpc and self.bnneck:
            loss = {
                "cmpcs_loss": losses.cmpc_loss(
                    self.projection, visual_embed_bn, textual_embed_bn, labels
                ),
                "global_align_loss": losses.global_align_loss(
                    visual_embed, textual_embed, labels, self.mixture
                ),
            }
            return loss
        if not self.cmpc and not self.bnneck:
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
                    self.mixture,
                    scale_pos=self.scale_pos,
                    scale_neg=self.scale_neg,
                ),
            }
            return loss
        return NotImplementedError


def make_loss_evaluator(cfg):
    return LossComputation(cfg)
