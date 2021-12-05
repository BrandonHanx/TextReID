import torch
import torch.nn as nn


class HybridGRU(nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        hidden_dim = cfg.MODEL.GRU.NUM_UNITS
        vocab_size = cfg.MODEL.GRU.VOCABULARY_SIZE
        embed_size = cfg.MODEL.GRU.EMBEDDING_SIZE
        num_layers = cfg.MODEL.GRU.NUM_LAYER
        drop_out = 1 - cfg.MODEL.GRU.DROPOUT_KEEP_PROB

        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)

        self.gru = nn.GRU(
            embed_size,
            hidden_dim,
            num_layers=num_layers,
            dropout=drop_out,
            bidirectional=True,
            bias=False,
        )
        self.out_channels = hidden_dim * 2

        embed_dim = cfg.MODEL.HBGRU.EMBED_DIM
        assert embed_dim == self.out_channels

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=cfg.MODEL.HBGRU.NUM_HEADS,
            dim_feedforward=cfg.MODEL.HBGRU.FF_DIM,
            dropout=0.5,
        )
        self.layers = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=cfg.MODEL.HBGRU.DEPTH,
            norm=nn.LayerNorm(embed_dim),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, 11, embed_dim))

        self.whole = cfg.MODEL.WHOLE

        self._init_weight()

    def resize_batch(self, attribute_embed, att_nums, max_num=10):
        start_idx, end_dix = 0, 0
        squeeze_att_embed = []
        mask = torch.zeros(att_nums.shape[0], max_num + 1).cuda()
        for i, num in enumerate(att_nums):
            end_dix = start_idx + num
            att_embed = attribute_embed[start_idx:end_dix]
            if num < max_num:
                pad = torch.zeros((max_num - num, self.out_channels)).cuda()
                squeeze_att_embed.append(torch.cat((att_embed, pad)))
            else:
                squeeze_att_embed.append(att_embed[:max_num])
            start_idx = end_dix
            mask[i, num + 1 :] = 1
        squeeze_att_embed = torch.stack(squeeze_att_embed, dim=0)
        return squeeze_att_embed, mask

    def forward(self, captions):
        text = torch.stack(
            [caption.text for caption in captions], dim=0
        ).squeeze()  # b x 70
        text_length = torch.stack(
            [caption.length for caption in captions], dim=0
        ).squeeze()

        attributes, att_nums = [], []
        for caption in captions:
            attribute = caption.get_field("attribute")
            attributes.append(attribute)
            att_nums.append(len(attribute))

        att_text = torch.cat([att.text for att in attributes], dim=0)  # ? x 25
        att_length = torch.cat([att.length for att in attributes], dim=0)
        att_nums = torch.tensor(att_nums).cuda()

        text_embed = self.embed(text)
        att_embed = self.embed(att_text)

        text_embed = self.gru_out(text_embed, text_length)  # b x d
        att_embed = self.gru_out(att_embed, att_length)  # ? x d
        att_embed, mask = self.resize_batch(att_embed, att_nums)  # b x 10 x d
        # cls_tokens = self.cls_token.expand(text_embed.shape[0], -1, -1)
        text_embed = torch.cat(
            (text_embed.unsqueeze(1), att_embed), dim=1
        )  # b x 12 x d
        text_embed = text_embed + self.pos_embed
        text_embed = text_embed.transpose(0, 1)  # 12 x b x d
        text_embed = self.layers(text_embed, src_key_padding_mask=mask.bool())
        if self.whole:
            return text_embed.transpose(0, 1), mask, att_nums
        return text_embed[0]  # B, D

    def gru_out(self, embed, text_length):

        _, idx_sort = torch.sort(text_length, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        embed_sort = embed.index_select(0, idx_sort)
        length_list = text_length[idx_sort]
        pack = nn.utils.rnn.pack_padded_sequence(
            embed_sort, length_list.cpu(), batch_first=True
        )

        gru_sort_out, _ = self.gru(pack)
        gru_sort_out, _ = nn.utils.rnn.pad_packed_sequence(
            gru_sort_out, batch_first=True
        )

        gru_out = gru_sort_out.index_select(0, idx_unsort)
        #         gru_out = gru_out[:, 0]
        gru_out, _ = torch.max(gru_out, dim=1)

        return gru_out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data, 1)
                nn.init.constant(m.bias.data, 0)


def build_hbgru(cfg):
    model = HybridGRU(cfg)
    return model
