import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .loss import make_loss_evaluator


class HungrianHead(nn.Module):
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
        self.patch_embed_layer = nn.Linear(visual_size, self.embed_size)
        self.att_embed_layer = nn.Linear(textual_size, self.embed_size)

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

    def get_assign_similarity(self, sim_matrix):
        att_match_idx, patch_match_idx = linear_sum_assignment(
            -sim_matrix.detach().cpu().numpy()
        )
        att_match_idx = torch.tensor(att_match_idx).cuda().long()
        patch_match_idx = torch.tensor(patch_match_idx).cuda().long()
        sim = sim_matrix[att_match_idx, patch_match_idx].mean()
        return sim, att_match_idx, patch_match_idx

    def get_topk_assign_similarity(self, sim_matrix, k=1):
        topk_sim = 0.0
        for _ in range(k):
            sim, att_match_idx, patch_match_idx = self.get_assign_similarity(sim_matrix)
            sim_matrix[att_match_idx, patch_match_idx] = -100.0
            topk_sim += sim
        return topk_sim / k

    def get_local_similarity(self, att_embed_all, patch_embed_all, att_nums):
        local_similarity = []
        start_idx, end_idx = 0, 0
        att_embed_all = F.normalize(att_embed_all, p=2, dim=-1)
        patch_embed_all = F.normalize(patch_embed_all, p=2, dim=-1)
        for patch_embed, num in zip(patch_embed_all, att_nums):
            end_idx = start_idx + num
            att_embed = att_embed_all[start_idx:end_idx]  # ? x d
            start_idx = end_idx
            sim_matrix = torch.matmul(att_embed, patch_embed.t())  # ? x 48
            sim = self.get_topk_assign_similarity(sim_matrix, k=4)
            local_similarity.append(sim)
        local_similarity = torch.stack(local_similarity).cuda()
        return local_similarity

    def forward(
        self, visual_feature, textual_feature, attribute_feature, att_nums, captions
    ):

        if self.avgpool is not None:
            visual_feature = self.avgpool(visual_feature)

        visual_embed, patch_embed = visual_feature[:, 0], visual_feature[:, 1:]
        visual_embed = self.visual_embed_layer(visual_embed)
        textual_embed = self.textual_embed_layer(textual_feature)  # b x d

        if self.training:
            patch_embed = self.patch_embed_layer(patch_embed)
            attribute_embed = self.att_embed_layer(attribute_feature)  # ? x d
            local_similarity = self.get_local_similarity(
                attribute_embed, patch_embed, att_nums
            )

            losses = self.loss_evaluator(
                visual_embed,
                textual_embed,
                local_similarity,
                captions,
            )
            return None, losses

        outputs = list()
        outputs.append(visual_embed)
        outputs.append(textual_embed)
        return outputs, None


def build_hungrian_head(cfg, visual_size, textual_size):
    model = HungrianHead(cfg, visual_size, textual_size)
    return model
