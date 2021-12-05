import random

import clip
import torch
import torch.nn as nn

from lib.utils.directory import load_vocab_dict


class GRU(nn.Module):
    def __init__(
        self,
        hidden_dim,
        vocab_size,
        embed_size,
        num_layers,
        drop_out,
        bidirectional,
        use_onehot,
        get_mask_label,
        cut_mix,
        random_delete,
        cut_neg,
    ):
        super().__init__()

        self.use_onehot = use_onehot
        self.get_mask_label = get_mask_label
        self.cut_mix = cut_mix
        self.random_delete = random_delete
        self.cut_neg = cut_neg

        # word embedding
        if use_onehot == "yes":
            self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        elif use_onehot == "dynamic_clip":
            model, _ = clip.load("ViT-B/32", device="cuda", jit=False)
            model = model.float()
            model = model.float().eval()
            for param in model.parameters():
                param.requires_grad = False
            del model.visual
            self.embed = model
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
            bidirectional=bidirectional,
            bias=False,
        )
        self.out_channels = hidden_dim * 2 if bidirectional else hidden_dim

        self._init_weight()

    @staticmethod
    def encode_text(model, text):
        x = model.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = x + model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = model.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x @ model.text_projection

        return x

    @staticmethod
    def _cut_mix(origin, length, p=0.5, k=4):
        if random.random() < p:
            cut = torch.min(length).long() // 2
            origin_idx = torch.arange(origin.shape[0]).reshape(-1, k)  # b/4 x 4
            shuffle_perm = torch.randperm(k)
            shuffle_idx = origin_idx[:, shuffle_perm].view(-1)  # b
            if random.random() < 0.5:
                prefix = origin[shuffle_idx]
                prefix = prefix[:, :cut]  # b x cut
                suffix = origin[:, cut:]  # b x (l - cut)
            else:
                prefix = origin[:, :cut]
                suffix = origin[shuffle_idx]
                suffix = suffix[:, cut:]
                length = length[shuffle_idx]
            cut_mix = torch.cat((prefix, suffix), dim=1)
            return cut_mix, length
        return origin, length

    @staticmethod
    def _cut_neg(origin, length, K=4):
        N = length.shape[0]
        cut = torch.min(length).long() * 2 // 3
        neg = []
        neg_length = []
        for offset in range(K, N - K + 1, K):
            #             if random.random() < 0.5:
            prefix = origin[:, :cut]
            suffix = torch.cat([origin] * 2, dim=0)
            suffix = suffix[offset : offset + N, cut:]
            length = torch.cat([length] * 2, dim=0)
            length = length[offset : offset + N]
            #             else:
            #                 prefix = torch.cat([origin] * 2, dim=0)
            #                 prefix = prefix[offset : offset + N, :cut]
            #                 suffix = origin[:, cut:]
            neg.append(torch.cat([prefix, suffix], dim=1))
            neg_length.append(length)
        return (
            torch.stack(neg, dim=0).flatten(0, 1),
            torch.stack(neg_length, dim=0).flatten(0, 1),
        )

    @staticmethod
    def _random_delete(origin, length, p=0.5):
        if random.random() < p:
            del_idx = random.randint(0, int(torch.min(length)))
            origin = torch.cat((origin[:, :del_idx], origin[:, del_idx + 1 :]), dim=1)
            length = length - 1
            if random.random() < 0.5:
                del_idx = random.randint(0, int(torch.min(length)))
                origin = torch.cat(
                    (origin[:, :del_idx], origin[:, del_idx + 1 :]), dim=1
                )
                length = length - 1
        return origin, length

    def forward(self, captions, equal_length=True):
        if self.use_onehot == "dynamic_clip":
            text = [caption.text for caption in captions]
            text_length = (
                torch.tensor([caption.length for caption in captions]).cuda().long()
            )
            text_length = text_length.clamp(max=75) + 2
            text = clip.tokenize(text, context_length=77).cuda()
            text = self.encode_text(self.embed, text)
        else:
            if equal_length:
                text = torch.stack([caption.text for caption in captions], dim=1)
                text_length = torch.stack(
                    [caption.length for caption in captions], dim=1
                )
            else:
                text = torch.cat([caption.text for caption in captions], dim=0)
                text_length = torch.cat([caption.length for caption in captions], dim=0)

            text_length = text_length.view(-1)
            text = text.view(-1, text.size(-1))  # b x l
            if self.cut_mix and self.training:
                text, text_length = self._cut_mix(text, text_length)
            if self.random_delete and self.training:
                text, text_length = self._random_delete(text, text_length)

            if not self.use_onehot == "yes":
                bs, length = text.shape[0], text.shape[-1]
                text = text.view(-1)  # bl
                text = self.vocab_dict[text].reshape(
                    bs, length, -1
                )  # b x l x vocab_size
            if self.embed is not None:
                text = self.embed(text)

        gru_out = self.gru_out(text, text_length)
        if self.cut_neg and self.training:
            neg, neg_length = self._cut_neg(text, text_length)
            neg_out = self.gru_out(neg, neg_length)
            gru_out, _ = torch.max(gru_out, dim=1)
            neg_out, _ = torch.max(neg_out, dim=1)
            return gru_out, neg_out
        if not self.get_mask_label:
            gru_out, _ = torch.max(gru_out, dim=1)
            return gru_out
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


def build_gru(cfg, bidirectional):
    use_onehot = cfg.MODEL.GRU.ONEHOT
    hidden_dim = cfg.MODEL.GRU.NUM_UNITS
    vocab_size = cfg.MODEL.GRU.VOCABULARY_SIZE
    embed_size = cfg.MODEL.GRU.EMBEDDING_SIZE
    num_layer = cfg.MODEL.GRU.NUM_LAYER
    drop_out = 1 - cfg.MODEL.GRU.DROPOUT_KEEP_PROB
    get_mask_label = cfg.MODEL.GRU.GET_MASK_LABEL
    cut_mix = cfg.MODEL.GRU.CUT_MIX
    random_delete = cfg.MODEL.GRU.RANDOM_DELETE
    cut_neg = cfg.MODEL.GRU.CUT_NEG

    model = GRU(
        hidden_dim,
        vocab_size,
        embed_size,
        num_layer,
        drop_out,
        bidirectional,
        use_onehot,
        get_mask_label,
        cut_mix,
        random_delete,
        cut_neg,
    )

    if cfg.MODEL.FREEZE:
        for m in [model.embed, model.gru]:
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    return model
