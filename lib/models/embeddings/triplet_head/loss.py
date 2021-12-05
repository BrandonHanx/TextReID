import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import lib.models.losses as losses


def triplet_loss(im, s, margin=0.2, max_violation=True):
    # compute image-sentence score matrix
    im = F.normalize(im, p=2, dim=-1)
    s = F.normalize(s, p=2, dim=-1)
    scores = torch.matmul(im, s.t())
    diagonal = scores.diag().view(im.size(0), 1)
    d1 = diagonal.expand_as(scores)
    d2 = diagonal.t().expand_as(scores)

    # compare every diagonal score to scores in its column
    # caption retrieval
    cost_s = (margin + scores - d1).clamp(min=0)
    # compare every diagonal score to scores in its row
    # image retrieval
    cost_im = (margin + scores - d2).clamp(min=0)

    # clear diagonals
    mask = torch.eye(scores.size(0)) > 0.5
    I = Variable(mask).cuda()
    cost_s = cost_s.masked_fill_(I, 0)
    cost_im = cost_im.masked_fill_(I, 0)

    # keep the maximum violating negative for each query
    if max_violation:
        cost_s = cost_s.max(1)[0]
        cost_im = cost_im.max(0)[0]

    return cost_s.sum() + cost_im.sum()


class LossComputation(nn.Module):
    def forward(
        self,
        visual_embed,
        textual_embed,
        captions,
    ):
        labels = torch.stack([caption.get_field("id") for caption in captions]).long()
        #         loss = {"triplet_loss": triplet_loss(visual_embed, textual_embed)}
        loss = {
            "align_loss": losses.global_align_loss(visual_embed, textual_embed, labels)
        }
        return loss


def make_loss_evaluator(cfg):
    return LossComputation()
