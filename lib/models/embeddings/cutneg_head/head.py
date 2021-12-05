import torch.nn as nn

from .loss import make_loss_evaluator


class CutNegHead(nn.Module):
    def __init__(
        self,
        cfg,
        visual_size,
        textual_size,
    ):
        super().__init__()
        self.embed_size = cfg.MODEL.EMBEDDING.FEATURE_SIZE

        self.visual_embed_layer = nn.Linear(visual_size, self.embed_size)
        self.textual_embed_layer = nn.Linear(textual_size, self.embed_size)

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

        visual_embed = self.visual_embed_layer(visual_feature)
        if self.training:
            textual_feature, neg_textual_feature = textual_feature
            textual_embed = self.textual_embed_layer(textual_feature)
            neg_textual_embed = self.textual_embed_layer(neg_textual_feature)
            losses = self.loss_evaluator(
                visual_embed, textual_embed, neg_textual_embed, captions
            )
            return None, losses
        else:
            textual_embed = self.textual_embed_layer(textual_feature)
            outputs = list()
            outputs.append(visual_embed)
            outputs.append(textual_embed)
            return outputs, None


def build_cutneg_head(cfg, visual_size, textual_size):
    model = CutNegHead(cfg, visual_size, textual_size)
    return model
