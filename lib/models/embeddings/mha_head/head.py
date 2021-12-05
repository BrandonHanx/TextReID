import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .loss import make_loss_evaluator


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k=None, d_v=None, temperature=20.0):
        super().__init__()

        self.n_head = n_head
        if d_k is None:
            d_k = d_model
        if d_v is None:
            d_v = d_model
        self.d_k, self.d_v = d_k, d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        # self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.w_vs = nn.Identity()

        self.temperature = temperature

    def scaled_dot_product_attention(self, q, k, v, k_mask=None):
        q = F.normalize(q, p=2, dim=-1)  # n x lq x dv
        k = F.normalize(k, p=2, dim=-1)  # bk x n x lk x dv
        attn = torch.matmul(q, k.transpose(2, 3))
        attn = F.relu(attn) * self.temperature  # bk x n x lq x lk
        if k_mask is not None:
            attn = attn.permute(1, 2, 0, 3)  # n x lq x bk x lk
            attn[:, :, k_mask == 1] = -1e9
            attn = attn.permute(2, 1, 0, 3)  # bk x n x lq x lk

        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v)  # bk x n x lq x dv

        return output

    def forward_qkv(self, q, k, v):
        return self.w_qs(q), self.w_ks(k), self.w_vs(v)

    def forward_attention(self, q, k, v, k_mask=None):
        bs_q, bs_k, bs_v = q.size(0), k.size(0), v.size(0)
        len_q, len_k, len_v = q.size(-2), k.size(-2), v.size(-2)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x n x lq x dv
        q = q.view(bs_q, self.n_head, len_q, self.d_k)
        k = k.view(bs_k, self.n_head, len_k, self.d_k)
        v = v.view(bs_v, self.n_head, len_v, self.d_v)

        att_qs = []
        for i in range(bs_q):
            # Broadcasting across batches
            att_q = self.scaled_dot_product_attention(q[i], k, v, k_mask)
            # Transpose to move the head dimension back: bk x lq x n x dv
            # Combine the last two dimensions to concatenate all the heads together: bk x lq x (n*dv)
            att_q = att_q.transpose(1, 2).contiguous().view(bs_k, len_q, -1)
            att_qs.append(att_q)

        return torch.stack(att_qs, dim=1)

    def forward(self, q, k, v, k_mask=None):
        q, k, v = self.forward_qkv(q, k, v)
        return self.forward_attention(q, k, v, k_mask)


class MHAHead(nn.Module):
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
        self.mha = MultiHeadAttention(d_model=self.embed_size, n_head=1)
        self.whole = cfg.MODEL.WHOLE

        self.loss_evaluator = make_loss_evaluator(cfg)
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode="fan_out")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_similarity(self, textual_embed, visual_embed, chunk_size=None, mask=None):
        if len(visual_embed.shape) < 3:
            visual_embed = visual_embed.unsqueeze(1)
        if len(textual_embed.shape) < 3:
            textual_embed = textual_embed.unsqueeze(1)

        textual_norm = F.normalize(textual_embed, p=2, dim=-1)
        visual_norm = F.normalize(visual_embed, p=2, dim=-1)

        if chunk_size is None:  # use huge GPU memory
            att_text_feat = self.mha(
                q=visual_norm, k=textual_norm, v=textual_embed, k_mask=mask
            )  # bt, bv, lv, d
            att_vis_feat = self.mha(
                q=textual_norm, k=visual_norm, v=visual_embed
            )  # bv, bt, lt, d
            att_text_feat = F.normalize(att_text_feat, p=2, dim=-1)
            att_vis_feat = F.normalize(att_vis_feat, p=2, dim=-1)
            if mask is not None:
                att_vis_feat[:, mask == 1, :] = 0
            similarity = torch.matmul(
                att_text_feat, att_vis_feat.permute(1, 0, 3, 2)
            )  # bt, bv, lt, lv
            similarity = similarity.mean(dim=(2, 3)).cuda()

            return similarity

        num_img = len(visual_norm)
        num_text = len(textual_norm)
        similarity = torch.zeros(num_text, num_img).cuda()
        for i in tqdm(range(num_text // chunk_size + 1)):
            ii = i * chunk_size
            for j in range(num_img // chunk_size + 1):
                jj = j * chunk_size
                att_text_feat = self.mha(
                    visual_norm[jj : jj + chunk_size],
                    textual_norm[ii : ii + chunk_size],
                    textual_embed[ii : ii + chunk_size],
                    mask[ii : ii + chunk_size] if mask else None,
                )
                att_vis_feat = self.mha(
                    textual_norm[ii : ii + chunk_size],
                    visual_norm[jj : jj + chunk_size],
                    visual_embed[jj : jj + chunk_size],
                )
                att_text_feat = F.normalize(att_text_feat, p=2, dim=-1)
                att_vis_feat = F.normalize(att_vis_feat, p=2, dim=-1)
                if mask is not None:
                    att_vis_feat[:, mask[ii : ii + chunk_size] == 1, :] = 0
                similarity[ii : ii + chunk_size, jj : jj + chunk_size] = torch.matmul(
                    att_text_feat, att_vis_feat.permute(1, 0, 3, 2)
                ).mean(dim=(2, 3))

        return similarity

    def forward(self, visual_feature, textual_feature, captions):

        if self.avgpool is not None:
            visual_feature = self.avgpool(visual_feature)

        text_mask = None
        if self.whole:
            captions, text_mask = captions

        visual_embed = self.visual_embed_layer(visual_feature)
        textual_embed = self.textual_embed_layer(textual_feature)

        if self.training:
            similarity = self.get_similarity(
                textual_embed, visual_embed, mask=text_mask
            )
            if self.whole:
                visual_embed = visual_embed.mean(dim=-2)
                textual_embed = textual_embed.mean(dim=-2)
            losses = self.loss_evaluator(
                visual_embed, textual_embed, similarity, captions
            )
            return None, losses

        outputs = list()
        outputs.append(visual_embed)
        outputs.append(textual_embed)
        if self.whole:
            outputs.append(text_mask)
        return outputs, None


def build_mha_head(cfg, visual_size, textual_size):
    model = MHAHead(cfg, visual_size, textual_size)
    return model
