import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from .loss import make_loss_evaluator


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class GRLHead(nn.Module):
    def __init__(
        self,
        cfg,
        visual_size,
        textual_size,
    ):
        super().__init__()
        self.embed_size = cfg.MODEL.EMBEDDING.FEATURE_SIZE

        if cfg.MODEL.VISUAL_MODEL.split("_")[0] == "vit":
            self.avgpool = None
        else:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.visual_embed_layer = nn.Linear(visual_size, self.embed_size)
        self.textual_embed_layer = nn.Linear(textual_size, self.embed_size)
        self.discriminator = nn.Sequential(
            #             nn.Linear(self.embed_size, 256),
            #             nn.ReLU(inplace=True),
            #             GradientReversalLayer(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 2),
        )

        self.loss_evaluator = make_loss_evaluator(cfg)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, visual_feature, textual_feature, captions):
        batch_size = visual_feature.size(0)

        if self.avgpool is not None:
            visual_feature = self.avgpool(visual_feature)

        visual_embed = visual_feature.view(batch_size, -1)
        textual_embed = textual_feature.view(batch_size, -1)
        visual_embed = self.visual_embed_layer(visual_embed)
        textual_embed = self.textual_embed_layer(textual_embed)

        if self.training:
            visual_norm = F.normalize(visual_embed, p=2, dim=1)
            textual_norm = F.normalize(textual_embed, p=2, dim=1)
            visual_domain_logits = self.discriminator(visual_norm)
            textual_domain_logits = self.discriminator(textual_norm)
            losses = self.loss_evaluator(
                visual_embed,
                textual_embed,
                captions,
                visual_domain_logits,
                textual_domain_logits,
            )
            return None, losses

        outputs = list()
        outputs.append(visual_embed)
        outputs.append(textual_embed)
        return outputs, None


def build_grl_head(cfg, visual_size, textual_size):
    model = GRLHead(cfg, visual_size, textual_size)
    return model
