import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils.directory import load_vocab_dict


class AttentionPool(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        output_dim=None,
    ):
        super().__init__()

        self.positional_embedding = nn.Parameter(
            torch.randn(101, embed_dim) / embed_dim ** 0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x, mask):
        x = x.transpose(0, 1)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        max_length = x.shape[0]
        x = x + self.positional_embedding[:max_length, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x,
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
            ),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
            key_padding_mask=mask,
        )

        return x[0]


class SAGRU(nn.Module):
    def __init__(
        self,
        hidden_dim,
        vocab_size,
        embed_size,
        num_layers,
        drop_out,
        use_onehot,
    ):
        super().__init__()

        self.use_onehot = use_onehot

        # word embedding
        if use_onehot == "yes":
            self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        else:
            if vocab_size == embed_size:
                self.embed = None
            else:
                self.embed = nn.Linear(vocab_size, embed_size)

            vocab_dict = load_vocab_dict(use_onehot)
            assert vocab_size == vocab_dict.shape[1]
            self.vocab_dict = torch.tensor(vocab_dict).cuda().float()

        self.gru = nn.GRU(
            embed_size,
            hidden_dim,
            num_layers=num_layers,
            dropout=drop_out,
            bidirectional=True,
            bias=False,
        )
        self.out_channels = hidden_dim
        self.attention_pool = AttentionPool(hidden_dim * 2, 8, hidden_dim)

        self._init_weight()

    @staticmethod
    def get_key_padding_mask(text_length):
        batch_size = text_length.shape[0]
        max_length = torch.max(text_length) + 1
        mask = torch.zeros(batch_size, max_length).cuda()
        for i, length in enumerate(text_length):
            mask[i, length + 1 :] = 1
        return mask.bool()

    def forward(self, captions):

        text = torch.cat([caption.text for caption in captions], dim=0)
        text_length = torch.cat([caption.length for caption in captions], dim=0)

        if not self.use_onehot == "yes":
            bs, length = text.shape[0], text.shape[-1]
            text = text.view(-1)  # bl
            text = self.vocab_dict[text].reshape(bs, length, -1)  # b x l x vocab_size
        if self.embed is not None:
            text = self.embed(text)

        gru_out = self.gru_out(text, text_length)
        mask = self.get_key_padding_mask(text_length)
        gru_out = self.attention_pool(gru_out, mask)

        return gru_out

    def gru_out(self, embed, text_length):

        _, idx_sort = torch.sort(text_length, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        embed_sort = embed.index_select(0, idx_sort)
        length_list = text_length[idx_sort]
        pack = nn.utils.rnn.pack_padded_sequence(
            embed_sort, length_list.cpu(), batch_first=True
        )

        gru_sort_out, _ = self.gru(pack)
        gru_sort_out = nn.utils.rnn.pad_packed_sequence(gru_sort_out, batch_first=True)
        gru_sort_out = gru_sort_out[0]

        gru_out = gru_sort_out.index_select(0, idx_unsort)
        return gru_out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data, 1)
                nn.init.constant(m.bias.data, 0)


def build_sagru(cfg):
    use_onehot = cfg.MODEL.GRU.ONEHOT
    hidden_dim = cfg.MODEL.GRU.NUM_UNITS
    vocab_size = cfg.MODEL.GRU.VOCABULARY_SIZE
    embed_size = cfg.MODEL.GRU.EMBEDDING_SIZE
    num_layer = cfg.MODEL.GRU.NUM_LAYER
    drop_out = 1 - cfg.MODEL.GRU.DROPOUT_KEEP_PROB

    model = SAGRU(
        hidden_dim,
        vocab_size,
        embed_size,
        num_layer,
        drop_out,
        use_onehot,
    )

    return model
