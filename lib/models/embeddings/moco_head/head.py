import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .loss import make_loss_evaluator


class MoCoHead(nn.Module):
    def __init__(
        self,
        cfg,
        visual_model,
        textual_model,
    ):
        super().__init__()
        self.embed_size = cfg.MODEL.EMBEDDING.FEATURE_SIZE
        self.K = cfg.MODEL.MOCO.K
        self.m = cfg.MODEL.MOCO.M
        self.fc = cfg.MODEL.MOCO.FC

        self.v_encoder_q = visual_model
        self.t_encoder_q = textual_model
        self.v_encoder_k = copy.deepcopy(visual_model)
        self.t_encoder_k = copy.deepcopy(textual_model)
        for param in self.v_encoder_k.parameters():
            param.requires_grad = False
        for param in self.t_encoder_k.parameters():
            param.requires_grad = False

        if self.fc:
            self.v_fc_q = nn.Sequential(
                nn.Linear(visual_model.out_channels, self.embed_size),
                nn.ReLU(),
                nn.Linear(self.embed_size, self.embed_size),
            )
            self.t_fc_q = nn.Sequential(
                nn.Linear(textual_model.out_channels, self.embed_size),
                nn.ReLU(),
                nn.Linear(self.embed_size, self.embed_size),
            )
            self.v_fc_k = copy.deepcopy(self.v_fc_q)
            self.t_fc_k = copy.deepcopy(self.t_fc_q)
            for param in self.v_fc_k.parameters():
                param.requires_grad = False
            for param in self.t_fc_k.parameters():
                param.requires_grad = False

        self.v_embed_layer = nn.Linear(visual_model.out_channels, self.embed_size)
        self.t_embed_layer = nn.Linear(textual_model.out_channels, self.embed_size)

        self.register_buffer("t_queue", torch.rand(self.embed_size, self.K))
        self.t_queue = F.normalize(self.t_queue, dim=0)
        self.register_buffer("v_queue", torch.rand(self.embed_size, self.K))
        self.v_queue = F.normalize(self.v_queue, dim=0)
        # initialize id label as -1
        self.register_buffer("id_queue", -torch.ones((1, self.K), dtype=torch.long))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

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

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.v_encoder_q.parameters(), self.v_encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
        for param_q, param_k in zip(
            self.t_encoder_q.parameters(), self.t_encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
        if self.fc:
            for param_q, param_k in zip(
                self.v_fc_q.parameters(), self.v_fc_k.parameters()
            ):
                param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
            for param_q, param_k in zip(
                self.t_fc_q.parameters(), self.t_fc_k.parameters()
            ):
                param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, v_keys, t_keys, id_keys):
        batch_size = v_keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.v_queue[:, ptr : ptr + batch_size] = v_keys.T
        self.t_queue[:, ptr : ptr + batch_size] = t_keys.T
        self.id_queue[:, ptr : ptr + batch_size] = id_keys.T

        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr

    def forward(self, images, captions):
        N = images.shape[0]

        v_embed = self.v_encoder_q(images)
        t_embed = self.t_encoder_q(captions)

        if self.training:
            if self.fc:
                v_embed_q = self.v_fc_q(v_embed)
                t_embed_q = self.t_fc_q(t_embed)
                v_embed = self.v_embed_layer(v_embed)
                t_embed = self.t_embed_layer(t_embed)
                v_embed_q = F.normalize(v_embed_q, dim=1)
                t_embed_q = F.normalize(t_embed_q, dim=1)
            else:
                v_embed = self.v_embed_layer(v_embed)
                t_embed = self.t_embed_layer(t_embed)
                v_embed_q = F.normalize(v_embed, dim=1)
                t_embed_q = F.normalize(t_embed, dim=1)
            id_q = torch.stack([caption.get_field("id") for caption in captions]).long()

            with torch.no_grad():
                self._momentum_update_key_encoder()
                v_embed_k = self.v_encoder_k(images)
                if self.fc:
                    v_embed_k = self.v_fc_k(v_embed_k)
                else:
                    v_embed_k = self.v_embed_layer(v_embed_k)
                v_embed_k = F.normalize(v_embed_k, dim=1)
                t_embed_k = self.t_encoder_k(captions)
                if self.fc:
                    t_embed_k = self.t_fc_k(t_embed_k)
                else:
                    t_embed_k = self.t_embed_layer(t_embed_k)
                t_embed_k = F.normalize(t_embed_k, dim=1)

            # regard same instance ids as positive sapmles, we need filter them out
            pos_idx = (
                self.id_queue.expand(N, self.K)
                .eq(id_q.unsqueeze(-1))
                .nonzero(as_tuple=False)[:, 1]
            )
            unique, counts = torch.unique(
                torch.cat([torch.arange(self.K).long().cuda(), pos_idx]),
                return_counts=True,
            )
            neg_idx = unique[counts == 1]

            # v positive logits: Nx1
            v_pos = torch.einsum("nc,nc->n", [v_embed_q, t_embed_k]).unsqueeze(-1)
            # v negative logits: NxK
            t_queue = self.t_queue.clone().detach()
            t_queue = t_queue[:, neg_idx]
            v_neg = torch.einsum("nc,ck->nk", [v_embed_q, t_queue])
            # t positive logits: Nx1
            t_pos = torch.einsum("nc,nc->n", [t_embed_q, v_embed_k]).unsqueeze(-1)
            # t negative logits: NxK
            v_queue = self.v_queue.clone().detach()
            v_queue = v_queue[:, neg_idx]
            t_neg = torch.einsum("nc,ck->nk", [t_embed_q, v_queue])

            losses = self.loss_evaluator(
                v_embed, t_embed, v_pos, v_neg, t_pos, t_neg, id_q
            )
            self._dequeue_and_enqueue(v_embed_k, t_embed_k, id_q)
            return losses

        v_embed = self.v_embed_layer(v_embed)
        t_embed = self.t_embed_layer(t_embed)
        outputs = list()
        outputs.append(v_embed)
        outputs.append(t_embed)
        return outputs


def build_moco_head(cfg, visual_model, textual_model):
    return MoCoHead(cfg, visual_model, textual_model)
