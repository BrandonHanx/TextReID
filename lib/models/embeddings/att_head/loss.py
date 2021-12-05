import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import lib.models.losses as losses


def bce(pred, label):
    return 10 * F.binary_cross_entropy_with_logits(pred, F.sigmoid(label))


class InstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        # Maybe we need mask same instances
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss


class ClusterLoss(nn.Module):
    def __init__(self, class_num, temperature, device):
        super(ClusterLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_clusters(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j):
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss + ne_loss


class LossComputation(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.epsilon = cfg.MODEL.EMBEDDING.EPSILON

        self.projection = Parameter(
            torch.randn(cfg.MODEL.EMBEDDING.FEATURE_SIZE, cfg.MODEL.NUM_CLASSES),
            requires_grad=True,
        )
        #         self.gender_projection = Parameter(
        #             torch.randn(cfg.MODEL.EMBEDDING.FEATURE_SIZE, 3),
        #             requires_grad=True,
        #         )
        #         self.js_projection = Parameter(
        #             torch.randn(cfg.MODEL.EMBEDDING.FEATURE_SIZE, 100),
        #             requires_grad=True,
        #         )
        #         self.gender_weight = torch.tensor([1.0, 1.1153, 8.4304]).cuda()
        #         self.instance_projector = nn.Sequential(
        #             nn.Linear(256, 256),
        #             nn.ReLU(),
        #             nn.Linear(256, 128),
        #         )
        self.cluster_projector = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 20), nn.Softmax(dim=1)
        )
        self.cluster_loss = ClusterLoss(20, 1.0, "cuda")
        self.instance_loss = InstanceLoss(cfg.SOLVER.IMS_PER_BATCH, 0.5, "cuda")
        nn.init.xavier_uniform_(self.projection.data, gain=1)

    #         nn.init.xavier_uniform_(self.gender_projection.data, gain=1)

    def forward(
        self,
        visual_embed,
        textual_embed,
        captions,
    ):
        labels = torch.stack([caption.get_field("id") for caption in captions]).long()
        #         gender_labels = torch.stack(
        #             [caption.get_field("gender") for caption in captions]
        #         ).long()
        loss = {
            "instance_loss": losses.instance_loss(
                self.projection,
                visual_embed,
                textual_embed,
                labels,
                epsilon=self.epsilon,
            ),
            "global_align_loss": losses.global_align_loss(
                visual_embed,
                textual_embed,
                labels,
            ),
            #             "gender_loss": losses.weighted_cross_entropy_loss(
            #                 self.gender_projection,
            #                 visual_embed,
            #                 textual_embed,
            #                 gender_labels,
            #                 self.gender_weight,
            #             ),
            #             "js_loss": self.instance_loss(F.normalize(self.instance_projector(visual_embed), dim=1), F.normalize(self.instance_projector(textual_embed), dim=1)),
            "cluster_loss": self.cluster_loss(
                self.cluster_projector(visual_embed),
                self.cluster_projector(textual_embed),
            ),
        }
        return loss


def make_loss_evaluator(cfg):
    return LossComputation(cfg)
