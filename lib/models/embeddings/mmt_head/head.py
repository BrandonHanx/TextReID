import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .loss import make_loss_evaluator


class MMTHead(nn.Module):
    def __init__(
        self,
        cfg,
        visual_size,
        textual_size,
    ):
        super().__init__()
        self.embed_dim = cfg.MODEL.EMBEDDING.FEATURE_SIZE
        self.inference_mode = cfg.MODEL.INFERENCE_MODE
        self.batch_size = cfg.SOLVER.IMS_PER_BATCH
        self.task = cfg.MODEL.EMBEDDING.TASK

        self.visual_embed_layer = nn.Linear(visual_size, self.embed_dim)
        self.proj_layer = nn.Linear(1024, self.embed_dim)
        self.textual_embed_layer = nn.Linear(textual_size, self.embed_dim)

        if cfg.MODEL.VISUAL_MODEL.split("_")[0] == "vit":
            self.avgpool = None
        elif cfg.MODEL.VISUAL_MODEL in [
            "hbvit",
            "pit",
            "swin",
            "m_resnet",
            "m_resnet50",
            "m_resnet101",
            "clip_resnet",
        ]:
            self.avgpool = None
        else:
            if cfg.MODEL.RESNET.ATTN_POOL:
                self.avgpool = None
            else:
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=cfg.MODEL.TRANS_ENCODER.NUM_HEADS,
            dim_feedforward=cfg.MODEL.TRANS_ENCODER.FF_DIM,
            dropout=cfg.MODEL.TRANS_ENCODER.DROPOUT,
        )
        self.mmt = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=cfg.MODEL.TRANS_ENCODER.DEPTH,
            norm=nn.LayerNorm(self.embed_dim),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.v_seg_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.t_seg_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        self.word_seq_length = 100
        self.patch_seq_length = 24 * 8
        self.pos_token = nn.Parameter(
            torch.zeros(
                1, self.word_seq_length + self.patch_seq_length + 1, self.embed_dim
            )
        )
        #         self.align_classifier = nn.Sequential(
        #             nn.Linear(self.embed_dim, 1),
        #             nn.Sigmoid()
        #         )
        self.align_classifier = nn.Linear(self.embed_dim, 2)

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

    def _generate_random_mask(self, size, prob, area=None):
        random_mask = torch.rand(size)
        random_mask = random_mask < prob
        random_mask = random_mask.float().cuda()
        if area is not None:
            random_mask = random_mask * area
        random_mask = 1 - random_mask
        check_idx = random_mask.sum(-1) == size[-1]
        if check_idx.any():
            random_mask = self._generate_random_mask(
                size, prob, area
            )  # at least one mask
        return random_mask

    def apply_random_mask(self, seq, prob, area=None):
        random_mask = self._generate_random_mask(
            (seq.shape[0], seq.shape[1]), prob, area
        )
        masked_seq = seq * random_mask.unsqueeze(-1)
        return masked_seq, ~random_mask.bool()

    @staticmethod
    def get_hard_positive_idx(labels, dist_mat):
        batch_size = labels.size(0)
        labels_ = (
            labels.expand(batch_size, batch_size)
            .eq(labels.expand(batch_size, batch_size).t())
            .float()
        )
        #         random_select = torch.rand((batch_size, batch_size)).cuda()
        #         labels_ = labels_ * random_select
        labels_ = labels_ * dist_mat
        return torch.argmin(labels_, dim=1)

    @staticmethod
    def get_hard_negative_idx(labels, dist_mat):
        batch_size = labels.size(0)
        labels_ = (
            labels.expand(batch_size, batch_size)
            .ne(labels.expand(batch_size, batch_size).t())
            .float()
        )
        #         random_select = torch.rand((batch_size, batch_size)).cuda()
        #         labels_ = labels_ * random_select
        labels_ = labels_ * dist_mat
        return torch.argmax(labels_, dim=1)

    @staticmethod
    def hard_example_mining(dist_mat, labels):
        assert len(dist_mat.size()) == 2
        assert dist_mat.size(0) == dist_mat.size(1)
        N = dist_mat.size(0)

        # shape [N, N]
        is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
        is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N]
        _, relative_p_idx = torch.min(
            dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True
        )
        # `dist_an` means distance(anchor, negative)
        # both `dist_an` and `relative_n_inds` with shape [N]
        _, relative_n_idx = torch.max(
            dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True
        )

        # shape [N, N]
        ind = (
            labels.new()
            .resize_as_(labels)
            .copy_(torch.arange(0, N).long())
            .unsqueeze(0)
            .expand(N, N)
        )
        # shape [N]
        p_idx = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_idx.data
        ).squeeze()
        n_idx = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_idx.data
        ).squeeze()

        return p_idx, n_idx

    def get_key_padding_mask(self, text_length):
        batch_size = text_length.shape[0]
        mask = torch.zeros(
            batch_size, self.patch_seq_length + self.word_seq_length + 1
        ).cuda()
        for i in range(batch_size):
            mask[i, text_length[i] + self.patch_seq_length + 1 :] = 1
        return mask.bool()

    def forward_mmt(self, patch_seq, word_seq, key_padding_mask):
        batch_size = patch_seq.shape[0]
        patch_seq = patch_seq + self.v_seg_token
        word_seq = word_seq + self.t_seg_token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        seq = torch.cat((cls_tokens, patch_seq, word_seq), dim=1)
        seq = seq + self.pos_token
        seq = self.mmt(seq.transpose(0, 1), src_key_padding_mask=key_padding_mask)
        return seq.transpose(0, 1)  # b x 24*8+100+1 x 512

    def forward_itm(self, patch_seq, word_seq, key_padding_mask, labels, dist_mat):
        #         pos_idx, neg_idx = self.hard_example_mining(dist_mat, labels)
        pos_idx = self.get_hard_positive_idx(labels, dist_mat)
        neg_idx = self.get_hard_negative_idx(labels, dist_mat)
        pos_patch_seq = torch.index_select(patch_seq, dim=0, index=pos_idx)
        neg_patch_seq = torch.index_select(patch_seq, dim=0, index=neg_idx)
        pos_score = self.align_classifier(
            self.forward_mmt(pos_patch_seq, word_seq, key_padding_mask)[:, 0]
        )
        neg_score = self.align_classifier(
            self.forward_mmt(neg_patch_seq, word_seq, key_padding_mask)[:, 0]
        )
        score = torch.cat((pos_score, neg_score), dim=0)  # 2b
        loss = self.loss_evaluator.forward_itm(score)
        return loss

    def get_similarity(self, patch_seqs, word_seqs, key_padding_mask, chunk_size=512):
        num_images, num_texts = patch_seqs.shape[0], word_seqs.shape[0]
        similarity = torch.zeros(num_images, num_texts).cuda()
        for i in tqdm(range(num_images)):
            sim_per_image = torch.zeros(num_texts).cuda()
            for j in range(num_texts // chunk_size + 1):
                patch_seq = patch_seqs[i].unsqueeze(0).expand(chunk_size, -1, -1)
                word_seq = word_seqs[j : j + chunk_size]
                mask = key_padding_mask[j : j + chunk_size]
                logits = self.align_classifier(
                    self.forward_mmt(patch_seq, word_seq, mask)[:, 0]
                )  # chunk_size
                sim = logits.squeeze()  # chunk_size
                sim_per_image[j : j + chunk_size] = sim
            similarity[i] = sim_per_image
        return similarity.t()

    def forward_mwm(self, patch_seq, word_seq, key_padding_mask):
        masked_word_seq, random_mask = self.apply_random_mask(
            word_seq,
            prob=0.15,
            area=~key_padding_mask[
                :, 1 + self.patch_seq_length :
            ],  # masked word cannot be padding token
        )
        output_seq = self.forward_mmt(patch_seq, masked_word_seq, key_padding_mask)[
            :, 1 + self.patch_seq_length :
        ]
        masked_word = word_seq[random_mask]
        predicted_word = output_seq[random_mask]  # b*num_masked_word x d
        loss = self.loss_evaluator.forward_mwm(predicted_word, masked_word)
        return loss

    def forward_mpm(self, patch_seq, word_seq, key_padding_mask):
        masked_patch_seq, random_mask = self.apply_random_mask(patch_seq, prob=0.1)
        output_seq = self.forward_mmt(masked_patch_seq, word_seq, key_padding_mask)[
            :, 1 : 1 + self.patch_seq_length
        ]
        masked_patch = patch_seq[random_mask]
        predicted_patch = output_seq[random_mask]  # b*num_masked_patch x d
        loss = self.loss_evaluator.forward_mpm(predicted_patch, masked_patch)
        return loss

    def forward_cmr(self, visual_embed, textual_embed, labels):
        return self.loss_evaluator.forward_cmr(visual_embed, textual_embed, labels)

    def forward(self, visual_feature, textual_feature, captions):
        if self.avgpool is not None:
            visual_feature = self.avgpool(visual_feature)

        visual_embed, patch_embed = visual_feature
        word_embed = textual_feature
        textual_embed, _ = torch.max(textual_feature, dim=1)

        visual_embed = self.visual_embed_layer(visual_embed)
        textual_embed = self.textual_embed_layer(textual_embed)
        patch_embed = self.proj_layer(patch_embed)
        word_embed = self.proj_layer(word_embed)
        valid_length = word_embed.shape[1]
        word_embed = torch.cat(
            (
                word_embed,
                torch.zeros(
                    word_embed.shape[0],
                    self.word_seq_length - valid_length,
                    self.embed_dim,
                ).cuda(),
            ),
            dim=1,
        )

        text_length = torch.stack([caption.length for caption in captions])
        key_padding_mask = self.get_key_padding_mask(text_length)

        if self.training:
            labels = torch.stack(
                [caption.get_field("id") for caption in captions]
            ).long()

            losses = dict()
            if "CMR" in self.task:
                losses.update(self.forward_cmr(visual_embed, textual_embed, labels))
            if "ITM" in self.task:
                visual_norm = F.normalize(visual_embed, p=2, dim=1)
                textual_norm = F.normalize(textual_embed, p=2, dim=1)
                dist_mat = torch.matmul(textual_norm, visual_norm.t())
                losses.update(
                    self.forward_itm(
                        patch_embed,
                        word_embed,
                        key_padding_mask,
                        labels,
                        dist_mat,
                    )
                )
            if "MPM" in self.task:
                losses.update(
                    self.forward_mpm(patch_embed, word_embed, key_padding_mask)
                )
            if "MWM" in self.task:
                losses.update(
                    self.forward_mwm(patch_embed, word_embed, key_padding_mask)
                )

            return None, losses

        outputs = list()
        if self.inference_mode == "common":
            outputs.append(visual_embed)
            outputs.append(textual_embed)
        elif self.inference_mode == "cross":
            outputs.append(patch_embed)
            outputs.append(word_embed)
            outputs.append(key_padding_mask)
        else:
            NotImplementedError

        return outputs, None


def build_mmt_head(cfg, visual_size, textual_size):
    model = MMTHead(cfg, visual_size, textual_size)
    return model
