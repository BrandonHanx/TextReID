import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(
        self, hidden_dim, vocab_size, embed_size, drop_out, bidirectional, use_onehot
    ):
        super(LSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.drop_out = drop_out
        self.bidirectional = bidirectional
        self.use_onehot = use_onehot

        # word embedding
        if use_onehot:
            self.embed = nn.Embedding(self.vocab_size, self.embed_size, padding_idx=0)

        self.lstm = nn.ModuleList()
        self.lstm.append(
            nn.LSTM(
                self.embed_size,
                self.hidden_dim,
                num_layers=1,
                dropout=self.drop_out,
                bidirectional=False,
                bias=False,
            )
        )
        self.out_channels = hidden_dim

        if self.bidirectional:
            self.lstm.append(
                nn.LSTM(
                    self.embed_size,
                    self.hidden_dim,
                    num_layers=1,
                    dropout=self.drop_out,
                    bidirectional=False,
                    bias=False,
                )
            )
            self.out_channels = hidden_dim * 2

        self._init_weight()

    def forward(self, captions):
        text = torch.stack([caption.text for caption in captions], dim=1)
        text_length = torch.stack([caption.length for caption in captions], dim=1)
        device = text.device

        batch_size = text.size(1)
        text_length = text_length.view(-1)
        if self.use_onehot:
            text = text.view(-1, text.size(-1))
            embed = self.embed(text)
        else:
            embed = (
                text.permute(1, 0, 2)
                if text.dim() == 3
                else text.view(-1, text.size(-2), text.size(-1))
            )

        # unidirectional lstm
        lstm_out = self.lstm_out(embed, text_length, 0)

        if self.bidirectional:
            index_reverse = list(range(embed.shape[0] - 1, -1, -1))
            index_reverse = torch.LongTensor(index_reverse).to(device)
            embed_reverse = embed.index_select(0, index_reverse)
            text_length_reverse = text_length.index_select(0, index_reverse)
            lstm_out_bidirection = self.lstm_out(embed_reverse, text_length_reverse, 1)
            lstm_out_bidirection_reverse = lstm_out_bidirection.index_select(
                0, index_reverse
            )
            lstm_out = torch.cat([lstm_out, lstm_out_bidirection_reverse], dim=2)

        lstm_out, _ = torch.max(lstm_out, dim=1)
        lstm_out = (
            lstm_out.view(-1, batch_size, lstm_out.size(-1)).unsqueeze(3).unsqueeze(3)
        )

        return lstm_out

    def lstm_out(self, embed, text_length, index):

        _, idx_sort = torch.sort(text_length, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        embed_sort = embed.index_select(0, idx_sort)
        length_list = text_length[idx_sort]
        pack = nn.utils.rnn.pack_padded_sequence(
            embed_sort, length_list.cpu(), batch_first=True
        )

        lstm_sort_out, _ = self.lstm[index](pack)
        lstm_sort_out = nn.utils.rnn.pad_packed_sequence(
            lstm_sort_out, batch_first=True
        )
        lstm_sort_out = lstm_sort_out[0]

        lstm_out = lstm_sort_out.index_select(0, idx_unsort)

        return lstm_out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data, 1)
                nn.init.constant(m.bias.data, 0)


def build_lstm(cfg, bidirectional):
    use_onehot = cfg.MODEL.LSTM.ONEHOT
    hidden_dim = cfg.MODEL.LSTM.NUM_UNITS
    vocab_size = cfg.MODEL.LSTM.VOCABULARY_SIZE
    embed_size = cfg.MODEL.LSTM.EMBEDDING_SIZE
    drop_out = 1 - cfg.MODEL.LSTM.DROPOUT_KEEP_PROB

    model = LSTM(
        hidden_dim, vocab_size, embed_size, drop_out, bidirectional, use_onehot
    )

    if cfg.MODEL.FREEZE:
        for m in [model.embed, model.lstm]:
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    return model
