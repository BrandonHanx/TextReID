import math

import torch
import torch.nn as nn

from lib.utils.directory import load_vocab_dict


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        embed_dim = cfg.MODEL.TRANS_ENCODER.EMBED_DIM
        vocab_size = cfg.MODEL.TRANS_ENCODER.VOCABULARY_SIZE
        self.learn_ps = cfg.MODEL.TRANS_ENCODER.LEARN_PS
        self.use_onehot = cfg.MODEL.TRANS_ENCODER.ONEHOT
        self.max_length = 100

        if self.use_onehot == "yes":
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        else:
            if vocab_size == embed_dim:
                self.embedding = None
            else:
                self.embedding = nn.Linear(vocab_size, embed_dim)

            vocab_dict = load_vocab_dict(self.use_onehot)
            assert vocab_size == vocab_dict.shape[1]
            self.vocab_dict = torch.tensor(vocab_dict).cuda().float()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=cfg.MODEL.TRANS_ENCODER.NUM_HEADS,
            dim_feedforward=cfg.MODEL.TRANS_ENCODER.FF_DIM,
            dropout=0.0,
        )
        self.layers = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=cfg.MODEL.TRANS_ENCODER.DEPTH,
            norm=nn.LayerNorm(embed_dim),
        )
        if self.learn_ps:
            self.pos_encoder = nn.Parameter(
                torch.zeros(1, self.max_length + 1, embed_dim)
            )
        else:
            self.pos_encoder = PositionalEncoding(embed_dim)
        self.out_channels = embed_dim
        self.whole = cfg.MODEL.WHOLE

    def forward(self, captions):
        text = torch.cat([caption.text for caption in captions], dim=0)
        text_length = torch.cat([caption.length for caption in captions], dim=0)
        batch_size = text.shape[0]

        if not self.use_onehot == "yes":
            length = text.shape[-1]
            text = text.view(-1)  # bl
            text = self.vocab_dict[text].reshape(
                batch_size, length, -1
            )  # b x l x vocab_size

        mask = torch.zeros(batch_size, self.max_length + 1).cuda()
        for i in range(batch_size):
            mask[i, text_length[i] + 1 :] = 1

        if self.embedding is not None:
            x = self.embedding(text)  # B, S, D
        else:
            x = text

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # B, S+1, D

        if self.learn_ps:
            x = x + self.pos_encoder
            x = x.transpose(0, 1)
        else:
            x = x.transpose(0, 1)  # S+1, B, D
            x = self.pos_encoder(x)
        x = self.layers(x, src_key_padding_mask=mask.bool())  # S+1, B, D
        if self.whole:
            return x.transpose(0, 1), mask
        return x[0]  # B, D


def build_trans_encoder(cfg):
    return TransEncoder(cfg)
