import torch
import torch.nn as nn
import torch.nn.functional as F


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(
            1, targets.unsqueeze(1).data.cpu(), 1
        )
        if self.use_gpu:
            targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


def instance_loss(
    projection, visual_embed, textual_embed, labels, scale=1, norm=False, epsilon=0.0
):
    if norm:
        visual_norm = F.normalize(visual_embed, p=2, dim=-1)
        textual_norm = F.normalize(textual_embed, p=2, dim=-1)
    else:
        visual_norm = visual_embed
        textual_norm = textual_embed
    projection_norm = F.normalize(projection, p=2, dim=0)

    visual_logits = scale * torch.matmul(visual_norm, projection_norm)
    textual_logits = scale * torch.matmul(textual_norm, projection_norm)

    if epsilon > 0:
        criterion = CrossEntropyLabelSmooth(num_classes=projection_norm.shape[1])
    else:
        criterion = nn.CrossEntropyLoss(reduction="mean")
    loss = criterion(visual_logits, labels) + criterion(textual_logits, labels)

    return loss


def cmpc_loss(projection, visual_embed, textual_embed, labels, verbose=False):
    """
    Cross-Modal Projection Classfication loss (CMPC)
    :param image_embeddings: Tensor with dtype torch.float32
    :param text_embeddings: Tensor with dtype torch.float32
    :param labels: Tensor with dtype torch.int32
    :return:
    """
    visual_norm = F.normalize(visual_embed, p=2, dim=1)
    textual_norm = F.normalize(textual_embed, p=2, dim=1)
    projection_norm = F.normalize(projection, p=2, dim=0)

    image_proj_text = (
        torch.sum(visual_embed * textual_norm, dim=1, keepdim=True) * textual_norm
    )
    text_proj_image = (
        torch.sum(textual_embed * visual_norm, dim=1, keepdim=True) * visual_norm
    )

    image_logits = torch.matmul(image_proj_text, projection_norm)
    text_logits = torch.matmul(text_proj_image, projection_norm)

    criterion = nn.CrossEntropyLoss(reduction="mean")
    loss = criterion(image_logits, labels) + criterion(text_logits, labels)

    # classification accuracy for observation
    if verbose:
        image_pred = torch.argmax(image_logits, dim=1)
        text_pred = torch.argmax(text_logits, dim=1)

        image_precision = torch.mean((image_pred == labels).float())
        text_precision = torch.mean((text_pred == labels).float())

        return loss, image_precision, text_precision
    return loss


def ce_mask_loss(seg_feat, masks, num_parts=5):
    masks = torch.stack(masks, dim=1)  # 5 * b * h/8 * w/8
    masks = masks.view(-1, masks.size(-2), masks.size(-1))  # 5b * h/8 * w/8

    loss = F.cross_entropy(seg_feat, masks.long(), reduction="none")
    loss = num_parts * loss.mean()
    return loss


def bce_mask_loss(seg_feat, masks, num_parts=5):
    loss = F.binary_cross_entropy_with_logits(seg_feat, masks, reduction="mean")
    return loss * num_parts


def global_align_loss(
    visual_embed,
    textual_embed,
    labels,
    mixture=False,
    alpha=0.6,
    beta=0.4,
    scale_pos=10,
    scale_neg=40,
):
    batch_size = labels.size(0)
    visual_norm = F.normalize(visual_embed, p=2, dim=1)
    textual_norm = F.normalize(textual_embed, p=2, dim=1)
    similarity = torch.matmul(visual_norm, textual_norm.t())
    labels_ = (
        labels.expand(batch_size, batch_size)
        .eq(labels.expand(batch_size, batch_size).t())
        .float()
    )

    pos_inds = labels_ == 1
    neg_inds = labels_ == 0
    loss_pos = torch.log(1 + torch.exp(-scale_pos * (similarity[pos_inds] - alpha)))
    loss_neg = torch.log(1 + torch.exp(scale_neg * (similarity[neg_inds] - beta)))
    loss = (loss_pos.sum() + loss_neg.sum()) * 2.0

    if mixture:
        margin = alpha - beta
        tmp = similarity
        tmp[neg_inds] = 1
        hard_v_pos, _ = torch.min(tmp, dim=1)
        hard_t_pos, _ = torch.min(tmp, dim=0)
        tmp = similarity
        tmp[pos_inds] = 0
        hard_v_neg, _ = torch.max(tmp, dim=1)
        hard_t_neg, _ = torch.max(tmp, dim=0)
        #         y = torch.ones_like(hard_v_neg)
        #         loss_v_dist = F.margin_ranking_loss(hard_v_neg, hard_v_pos, y, margin=margin, reduction="sum")
        #         loss_t_dist = F.margin_ranking_loss(hard_t_neg, hard_t_pos, y, margin=margin, reduction="sum")
        v_dist = hard_v_pos - hard_v_neg
        t_dist = hard_t_pos - hard_t_neg
        loss_v_dist = torch.log(1 + torch.exp(margin - v_dist))
        loss_t_dist = torch.log(1 + torch.exp(margin - t_dist))
        loss = loss + loss_t_dist.sum() + loss_v_dist.sum()

    loss /= batch_size
    return loss


def global_align_loss_from_sim(
    similarity,
    labels,
    alpha=0.6,
    beta=0.4,
    scale_pos=10,
    scale_neg=40,
):
    batch_size = labels.size(0)
    labels_ = (
        labels.expand(batch_size, batch_size)
        .eq(labels.expand(batch_size, batch_size).t())
        .float()
    )

    pos_inds = labels_ == 1
    neg_inds = labels_ == 0
    loss_pos = torch.log(1 + torch.exp(-scale_pos * (similarity[pos_inds] - alpha)))
    loss_neg = torch.log(1 + torch.exp(scale_neg * (similarity[neg_inds] - beta)))
    loss = (loss_pos.sum() + loss_neg.sum()) * 2.0

    loss /= batch_size
    return loss


def cmpm_loss(visual_embed, textual_embed, labels, verbose=False, epsilon=1e-8):
    """
    Cross-Modal Projection Matching Loss(CMPM)
    :param image_embeddings: Tensor with dtype torch.float32
    :param text_embeddings: Tensor with dtype torch.float32
    :param labels: Tensor with dtype torch.int32
    :return:
        i2t_loss: cmpm loss for image projected to text
        t2i_loss: cmpm loss for text projected to image
        pos_avg_sim: average cosine-similarity for positive pairs
        neg_avg_sim: averate cosine-similarity for negative pairs
    """

    batch_size = visual_embed.shape[0]
    labels_reshape = torch.reshape(labels, (batch_size, 1))
    labels_dist = labels_reshape - labels_reshape.t()
    labels_mask = labels_dist == 0

    visual_norm = F.normalize(visual_embed, p=2, dim=1)
    textual_norm = F.normalize(textual_embed, p=2, dim=1)
    image_proj_text = torch.matmul(visual_embed, textual_norm.t())
    text_proj_image = torch.matmul(textual_embed, visual_norm.t())

    # normalize the true matching distribution
    labels_mask_norm = labels_mask.float() / labels_mask.float().norm(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (
        F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm + epsilon)
    )

    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (
        F.log_softmax(text_proj_image, dim=1) - torch.log(labels_mask_norm + epsilon)
    )

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(
        torch.sum(t2i_loss, dim=1)
    )

    if verbose:
        sim_cos = torch.matmul(visual_norm, textual_norm.t())

        pos_avg_sim = torch.mean(torch.masked_select(sim_cos, labels_mask))
        neg_avg_sim = torch.mean(torch.masked_select(sim_cos, labels_mask == 0))

        return loss, pos_avg_sim, neg_avg_sim
    return loss


def local_align_no_sampling_loss(
    part_embed,
    attr_embed,
    labels,
    part_masks,
    attr_masks,
    num_parts=5,
    alpha=0.6,
    beta=0.4,
    scale_pos=10,
    scale_neg=40,
):
    batch_size = labels.size(0)
    part_embed = F.normalize(part_embed, p=2, dim=2)
    attr_embed = F.normalize(attr_embed, p=2, dim=2)
    labels_ = labels.expand(batch_size, batch_size).eq(
        labels.expand(batch_size, batch_size).t()
    )

    pos_inds = labels_ == 1
    neg_inds = labels_ == 0

    local_loss = 0.0
    for i in range(num_parts):
        filter_inds = torch.ones_like(labels_)
        filter_inds[~attr_masks[:, i], :] = 0
        filter_inds[:, ~part_masks[:, i]] = 0
        filter_pos_inds = filter_inds & pos_inds
        filter_neg_inds = filter_inds & neg_inds

        local_similarity = torch.matmul(attr_embed[i], part_embed[i].t())
        loss_pos = torch.log(
            1 + torch.exp(-scale_pos * (local_similarity[filter_pos_inds] - alpha))
        )
        loss_neg = torch.log(
            1 + torch.exp(scale_neg * (local_similarity[filter_neg_inds] - beta))
        )
        local_loss += (loss_pos.sum() + loss_neg.sum()) * 2.0
    return local_loss / batch_size / num_parts


def local_align_loss(
    part_embed,
    attribute_embed,
    labels,
    part_masks,
    attr_masks,
    num_parts=5,
    alpha=0.6,
    beta=0.4,
    scale_pos=10,
    scale_neg=40,
    topK=8,
):

    batch_size = labels.size(0)
    part_embed = F.normalize(part_embed, p=2, dim=2)
    attribute_embed = F.normalize(attribute_embed, p=2, dim=2)
    labels_ = labels.expand(batch_size, batch_size).eq(
        labels.expand(batch_size, batch_size).t()
    )

    losses = 0
    for i in range(num_parts):
        part_mask = part_masks[:, i]
        attr_mask = attr_masks[:, i]
        similarity = torch.matmul(part_embed[i], attribute_embed[i].t())
        rank1 = torch.argsort(similarity, dim=1, descending=True)
        rank2 = torch.argsort(similarity.t(), dim=1, descending=True)

        loss = 0
        for j in range(batch_size):
            if part_mask[j] == 0:
                continue
            pred = similarity[j, attr_mask]
            # k-reciprocal sample
            label = labels_[j, :].float()
            forward_k_idx = rank1[i, :topK]
            backward_k_idx = rank2[forward_k_idx, :topK]
            sample_pos_idx = torch.nonzero(backward_k_idx == i)[:, 0]
            sample_pos_idx = torch.unique(forward_k_idx[sample_pos_idx])
            label[sample_pos_idx] = 1
            label = label[attr_mask]
            pos_inds = torch.nonzero(label == 1).squeeze(1)
            neg_inds = torch.nonzero(label == 0).squeeze(1)
            if pos_inds.numel() > 0:
                loss_pos = torch.log(
                    1 + torch.exp(-scale_pos * (pred[pos_inds] - alpha))
                )
                loss += loss_pos.sum()
            if neg_inds.numel() > 0:
                loss_neg = torch.log(1 + torch.exp(scale_neg * (pred[neg_inds] - beta)))
                loss += loss_neg.sum()

            if attr_mask[j] == 0:
                continue
            pred = similarity[part_mask, j]
            # k-reciprocal sample
            label = labels_[j, :].float()
            forward_k_idx = rank2[i, :topK]
            backward_k_idx = rank1[forward_k_idx, :topK]
            sample_pos_idx = torch.nonzero(backward_k_idx == i)[:, 0]
            sample_pos_idx = torch.unique(forward_k_idx[sample_pos_idx])
            label[sample_pos_idx] = 1
            label = label[part_mask]
            pos_inds = torch.nonzero(label == 1).squeeze(1)
            neg_inds = torch.nonzero(label == 0).squeeze(1)
            if pos_inds.numel() > 0:
                loss_pos = torch.log(
                    1 + torch.exp(-scale_pos * (pred[pos_inds] - alpha))
                )
                loss += loss_pos.sum()
            if neg_inds.numel() > 0:
                loss_neg = torch.log(1 + torch.exp(scale_neg * (pred[neg_inds] - beta)))
                loss += loss_neg.sum()

        loss /= batch_size
        losses += loss
    losses /= num_parts
    return losses


def domain_loss(visual_domain_logits, textual_domain_logits):
    criterion = nn.CrossEntropyLoss()
    batch_size = visual_domain_logits.shape[0]
    visual_domain_labels = torch.zeros(batch_size).long().cuda()
    textual_domain_labels = torch.ones(batch_size).long().cuda()
    loss = criterion(visual_domain_logits, visual_domain_labels) + criterion(
        textual_domain_logits, textual_domain_labels
    )
    return loss


def coral_loss(source, target):
    d = source.data.shape[1]

    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = xm.t() @ xm

    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = xmt.t() @ xmt

    # frobenius norm between source and target
    loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
    loss = loss / (4 * d * d)

    return loss


def weighted_cross_entropy_loss(projection, visual_embed, textual_embed, label, weight):
    visual_embed = visual_embed @ projection
    textual_embed = textual_embed @ projection
    loss = F.cross_entropy(visual_embed, label, weight=weight) + F.cross_entropy(
        textual_embed, label, weight=weight
    )
    return loss
