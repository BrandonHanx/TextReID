import torch
import torch.nn as nn
import torch.nn.functional as F

from .loss import make_loss_evaluator


class DirectHBHead(nn.Module):
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

    @staticmethod
    def focal_equal(attn, sourceL):
        """
        consider the confidence g(x) for each fragment as equal
        sigma_{j} (xi - xj) = sigma_{j} xi - sigma_{j} xj
        attn: (batch, queryL, sourceL)
        """
        funcF = attn * sourceL - torch.sum(attn, dim=-1, keepdim=True)
        fattn = torch.where(funcF > 0, torch.ones_like(attn), torch.zeros_like(attn))
        return fattn

    def scaled_dot_product_attention(self, q, k, v, k_mask=None):
        q = F.normalize(q, p=2, dim=-1)  # lq x dv
        k = F.normalize(k, p=2, dim=-1)  # bk x lk x dv
        attn = torch.matmul(q, k.transpose(1, 2))
        attn = F.relu(attn) * 20.0  # bk x lq x lk
        if k_mask is not None:
            attn = attn.transpose(0, 1)  # lq x bk x lk
            attn[:, k_mask == 1] = -1e9
            attn = attn.transpose(0, 1)  # bk x lq x lk
        attn = F.softmax(attn, dim=-1)

        #         focal_filter = self.focal_equal(attn, k.shape[1])
        #         tmp_attn = focal_filter * attn
        #         attn_sum = torch.sum(tmp_attn, dim=-1, keepdim=True)  # bk x lq x 1
        #         attn = tmp_attn / attn_sum  # bk x lq x lk

        output = torch.matmul(attn, v)  # bk x lq x dv

        return output

    def get_att_similarity(
        self, att_embed, patch_embed, chunk_size=None, att_mask=None, att_nums=None
    ):
        similarity = []
        attend_patch_embed = []
        for i in range(att_embed.shape[0]):
            attend_patch_embed.append(
                self.scaled_dot_product_attention(
                    att_embed[i], patch_embed, patch_embed
                )
            )  # bp x 10 x d
        attend_patch_embed = torch.stack(attend_patch_embed, dim=1)  # bp x ba x 10 x d
        if att_mask is not None:
            attend_patch_embed[:, att_mask.bool(), :] = 0

        attend_att_embed = []
        for i in range(patch_embed.shape[0]):
            attend_att_embed.append(
                self.scaled_dot_product_attention(
                    patch_embed[i], att_embed, att_embed, att_mask
                )
            )  # ba x 48 x d
        attend_att_embed = torch.stack(attend_att_embed, dim=1)  # ba x bp x 48 x d

        attend_patch_embed = F.normalize(attend_patch_embed, p=2, dim=-1)
        attend_att_embed = F.normalize(attend_att_embed, p=2, dim=-1)
        similarity = torch.matmul(
            attend_att_embed, attend_patch_embed.permute(1, 0, 3, 2)
        )

        if att_nums is not None:
            similarity /= att_nums.view(-1, 1, 1, 1)
            similarity = similarity.sum(dim=2).mean(dim=2)
        else:
            similarity = similarity.mean(dim=(2, 3))
        return similarity

    def forward(self, visual_feature, textual_feature, mask, att_nums, captions):

        if self.avgpool is not None:
            visual_feature = self.avgpool(visual_feature)

        visual_embed, patch_embed = visual_feature[:, 0], visual_feature[:, 1:]
        visual_embed = self.visual_embed_layer(visual_embed)
        patch_embed = self.patch_embed_layer(patch_embed)

        textual_embed, squeeze_att_embed = textual_feature[:, 0], textual_feature[:, 1:]
        textual_embed = self.textual_embed_layer(textual_embed)  # b x d
        squeeze_att_embed = self.att_embed_layer(squeeze_att_embed)

        if self.training:
            local_similarity = self.get_att_similarity(
                squeeze_att_embed, patch_embed, att_mask=mask[:, 1:], att_nums=att_nums
            )
            patch_embed = patch_embed.mean(dim=1)
            squeeze_att_embed = squeeze_att_embed / att_nums.view(-1, 1, 1)
            squeeze_att_embed = squeeze_att_embed.sum(dim=1)
            losses = self.loss_evaluator(
                visual_embed,
                textual_embed,
                patch_embed,
                squeeze_att_embed,
                local_similarity,
                captions,
            )
            return None, losses

        outputs = list()
        outputs.append(visual_embed)
        outputs.append(textual_embed)
        outputs.append(att_nums)
        return outputs, None


def build_direct_hb_head(cfg, visual_size, textual_size):
    model = DirectHBHead(cfg, visual_size, textual_size)
    return model
