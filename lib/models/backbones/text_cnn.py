import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        n_filters,
        filter_sizes,
        dropout,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=1,
                    out_channels=n_filters,
                    kernel_size=(fs, embed_dim),
                )
                for fs in filter_sizes
            ]
        )

        # self.fc = nn.Linear(len(filter_sizes) * n_filters, embed_dim)
        # self.out_channels = len(filter_sizes) * n_filters

        # self.dropout = nn.Dropout(dropout)

        self.pos_embed = nn.Parameter(torch.zeros(1, len(filter_sizes) + 1, n_filters))
        self.trans_layers = nn.ModuleList(
            [Block(dim=n_filters, num_heads=8) for i in range(4)]
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, n_filters))
        self.norm = nn.LayerNorm(n_filters)
        self.out_channels = n_filters

    def forward(self, captions):

        text = torch.stack([caption.text for caption in captions], dim=1)
        text = text.view(-1, text.size(-1))

        # text = [batch size, sent len]

        embedded = self.embedding(text)

        # embedded = [batch size, sent len, emb dim]

        embedded = embedded.unsqueeze(1)

        # embedded = [batch size, 1, sent len, emb dim]

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        # conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        # pooled_n = [batch size, n_filters]

        # cat = self.dropout(torch.cat(pooled, dim=1))
        cat = torch.cat(pooled, dim=1)

        # cat = [batch size, n_filters * len(filter_sizes)]

        cls_tokens = self.cls_token.expand(text.shape[0], -1, -1)
        x = torch.cat((cls_tokens, cat), dim=1)

        x = x + self.pos_embed

        for blk in self.trans_layers:
            x = blk(x)

        x = self.norm(x)

        return x[:, 0]


def build_text_cnn(cfg):
    cnn = TextCNN(
        cfg.MODEL.TEXT_CNN.VOCABULARY_SIZE,
        cfg.MODEL.TEXT_CNN.EMBEDDING_SIZE,
        cfg.MODEL.TEXT_CNN.NUM_FILTERS,
        cfg.MODEL.TEXT_CNN.FILTER_SIZE,
        cfg.MODEL.TEXT_CNN.DROPOUT,
    )
    return cnn
