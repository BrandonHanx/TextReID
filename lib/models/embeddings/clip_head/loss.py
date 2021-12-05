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

    #         self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def forward(
        self,
        visual_embed,
        textual_embed,
        captions,
    ):
        labels = torch.stack([caption.get_field("id") for caption in captions]).long()
        #         batch_size = labels.size(0)
        #         labels_ = (
        #             labels.expand(batch_size, batch_size)
        #             .eq(labels.expand(batch_size, batch_size).t())
        #             .float()
        #         )
        loss = {
            #             "instance_loss": losses.instance_loss(
            #                 self.projection, visual_embed, textual_embed, labels
            #             ),
            "global_align_loss": losses.global_align_loss(
                visual_embed, textual_embed, labels
            ),
        }
        return loss


#     def clip_loss(self, image_features, text_features, labels):
#         # normalized features
#         image_features = image_features / image_features.norm(dim=-1, keepdim=True)
#         text_features = text_features / text_features.norm(dim=-1, keepdim=True)

#         # cosine similarity as logits
#         logit_scale = self.logit_scale.exp()
#         logits_per_image = logit_scale * image_features @ text_features.t()
#         logits_per_text = logit_scale * text_features @ image_features.t()

#         criterion = nn.BCELoss()
#         loss = criterion(logits_per_image, labels) + criterion(logits_per_text, labels)
#         return loss / 2.0


def make_loss_evaluator(cfg):
    return LossComputation(cfg)
