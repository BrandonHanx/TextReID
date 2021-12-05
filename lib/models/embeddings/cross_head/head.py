import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .loss import make_loss_evaluator


class CrossHead(nn.Module):
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
        elif cfg.MODEL.VISUAL_MODEL == "hbvit":
            self.avgpool = None
        else:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

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

    @staticmethod
    def get_similarity_(textual_embed, visual_embed, att_mask=None):
        similarity = []
        for i in range(textual_embed.shape[0]):
            similarity.append(
                torch.matmul(textual_embed[i], visual_embed.transpose(1, 2))
            )
        similarity = torch.stack(similarity, dim=0)  # b x b x 11 x 49
        if att_mask is not None:
            similarity /= att_mask.view(-1, 1, 1, 1)
            similarity = similarity.sum(dim=2).mean(dim=2)
        else:
            similarity = similarity.mean(dim=(2, 3))
        return similarity

    def get_similarity(
        self, textual_embed, visual_embed, chunk_size=None, att_mask=None
    ):
        textual_embed = F.normalize(textual_embed, p=2, dim=-1)  # b x 11 x d
        visual_embed = F.normalize(visual_embed, p=2, dim=-1)  # b x 49 x d
        if chunk_size is None:
            return self.get_similarity_(textual_embed, visual_embed, att_mask)
        num_img = len(visual_embed)
        num_text = len(textual_embed)
        similarity = torch.zeros(num_text, num_img).cuda()
        for i in tqdm(range(num_text // chunk_size + 1)):
            ii = i * chunk_size
            for j in range(num_img // chunk_size + 1):
                jj = j * chunk_size
                similarity[
                    ii : ii + chunk_size, jj : jj + chunk_size
                ] = self.get_similarity_(
                    textual_embed[ii : ii + chunk_size],
                    visual_embed[jj : jj + chunk_size],
                    att_mask[ii : ii + chunk_size],
                )
        return similarity

    def forward(
        self, visual_feature, textual_feature, attribute_feature, att_nums, captions
    ):

        if self.avgpool is not None:
            visual_feature = self.avgpool(visual_feature)

        visual_embed = self.visual_embed_layer(visual_feature)
        textual_embed = self.textual_embed_layer(textual_feature)  # b x d
        attribute_embed = self.textual_embed_layer(attribute_feature)
        start_idx, end_dix = 0, 0
        max_num = 10
        squeeze_att_embed = []
        for num in att_nums:
            end_dix = start_idx + num
            att_embed = attribute_embed[start_idx:end_dix]
            if num < max_num:
                pad = torch.zeros((max_num - num, self.embed_size)).cuda()
                squeeze_att_embed.append(torch.cat((att_embed, pad)))
            else:
                squeeze_att_embed.append(att_embed[:max_num])
            start_idx = end_dix
        squeeze_att_embed = torch.stack(squeeze_att_embed, dim=0)
        textual_embed = torch.cat(
            (textual_embed.unsqueeze(1), squeeze_att_embed), dim=1
        )  # b x 11 x d
        att_mask = torch.tensor(att_nums).cuda() + 1

        if self.training:
            similarity = self.get_similarity(
                textual_embed, visual_embed, att_mask=att_mask
            )
            visual_embed = visual_embed.mean(dim=1)
            textual_embed = textual_embed.mean(dim=1)
            losses = self.loss_evaluator(
                visual_embed, textual_embed, similarity, captions
            )
            return None, losses

        outputs = list()
        outputs.append(visual_embed)
        outputs.append(textual_embed)
        outputs.append(att_mask)
        return outputs, None


def build_cross_head(cfg, visual_size, textual_size):
    model = CrossHead(cfg, visual_size, textual_size)
    return model
