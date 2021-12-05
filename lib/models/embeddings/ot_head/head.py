import torch
import torch.nn as nn
from tqdm import tqdm

from .loss import make_loss_evaluator
from .ot import optimal_transport_dist


class OTHead(nn.Module):
    def __init__(
        self,
        cfg,
        visual_size,
        textual_size,
    ):
        super().__init__()
        self.embed_size = cfg.MODEL.EMBEDDING.FEATURE_SIZE
        self.inference_mode = cfg.MODEL.INFERENCE_MODE

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
    def _compute_pad(lens):
        pad = torch.zeros(len(lens), max(lens), dtype=torch.uint8, device="cuda")
        for i, l in enumerate(lens):
            pad.data[i, l:].fill_(1)
        return pad.bool()

    @staticmethod
    def _pad_to_same(word_embed, txt_pad, max_length=100):
        valid_length = word_embed.shape[1]
        word_embed = torch.cat(
            (
                word_embed,
                torch.zeros(
                    word_embed.shape[0],
                    max_length - valid_length,
                    word_embed.shape[-1],
                    device="cuda",
                ),
            ),
            dim=1,
        )
        txt_pad = torch.cat(
            (
                txt_pad,
                torch.ones(
                    word_embed.shape[0],
                    max_length - valid_length,
                    device="cuda",
                ).bool(),
            ),
            dim=1,
        )
        return word_embed, txt_pad

    def get_similarity(self, patch_embed, word_embed, txt_pad, chunk_size=None):
        dist_mat = []
        num_txt = word_embed.shape[0]
        num_img = patch_embed.shape[0]
        img_pad = (
            torch.zeros((patch_embed.shape[0], patch_embed.shape[1])).cuda().bool()
        )
        if self.training:
            bar = range(num_txt)
        else:
            bar = tqdm(range(num_txt))
        for i in bar:
            # broadcast mannually
            word = word_embed[i]
            word = torch.stack([word] * num_img, dim=0)  # bv * sv * d
            pad = txt_pad[i]
            pad = torch.stack([pad] * num_img, dim=0)
            dist = optimal_transport_dist(word, patch_embed, pad, img_pad)  # bv
            dist_mat.append(dist)
        dist_mat = 1 - torch.stack(dist_mat, dim=0)  # bt * bv
        return dist_mat

    def forward(self, visual_feature, word_embed, captions):
        visual_embed, patch_embed = visual_feature
        textual_embed, _ = torch.max(word_embed, dim=1)

        patch_embed = self.visual_embed_layer(patch_embed)
        word_embed = self.textual_embed_layer(word_embed)
        visual_embed = self.visual_embed_layer(visual_embed)
        textual_embed = self.textual_embed_layer(textual_embed)

        txt_lens = torch.stack([caption.length for caption in captions])
        txt_pad = self._compute_pad(txt_lens)

        if self.training:
            labels = torch.stack(
                [caption.get_field("id") for caption in captions]
            ).long()
            dist_mat = self.get_similarity(patch_embed, word_embed, txt_pad)
            losses = self.loss_evaluator(visual_embed, textual_embed, dist_mat, labels)
            return None, losses

        outputs = list()
        word_embed, txt_pad = self._pad_to_same(word_embed, txt_pad)
        outputs.append(patch_embed)
        outputs.append(word_embed)
        outputs.append(txt_pad)
        return outputs, None


def build_ot_head(cfg, visual_size, textual_size):
    model = OTHead(cfg, visual_size, textual_size)
    return model
