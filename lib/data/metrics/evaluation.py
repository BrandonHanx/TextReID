import logging
import os

import numpy as np
import torch
import torch.nn.functional as F

from lib.utils.logger import table_log


def rank(similarity, q_pids, g_pids, topk=[1, 5, 10], get_mAP=True):
    max_rank = max(topk)
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        # acclerate sort with topk
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )  # q * topk
    pred_labels = g_pids[indices]  # q * k
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

    all_cmc = matches[:, :max_rank].cumsum(1)
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    all_cmc = all_cmc[topk - 1]

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k
    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100
    return all_cmc, mAP, indices


def jaccard(a_list, b_list):
    return float(len(set(a_list) & set(b_list))) / float(len(set(a_list) | set(b_list)))


def jaccard_mat(row_nn, col_nn, shape):
    jaccard_sim = np.zeros(shape)
    # FIXME: need optimization
    for i in range(min(shape[0], row_nn.shape[0])):
        for j in range(min(shape[1], col_nn.shape[0])):
            jaccard_sim[i, j] = jaccard(row_nn[i], col_nn[j])
    return jaccard_sim


def visual_neighbor_mat(single_feat, cross_sim, neighbor_num=5):
    cross_sim = cross_sim.t().cpu().numpy()  # v * t
    single_sim = torch.matmul(single_feat, single_feat.t()).cpu().numpy()  # v * v

    cross_indices = np.argsort(-cross_sim, axis=1)
    single_indices = np.argsort(-single_sim, axis=1)
    cross_nn = cross_indices[:, :neighbor_num]  # v * n
    single_nn = single_indices[:, :neighbor_num]  # v * n

    jaccard_sim = jaccard_mat(cross_nn, single_nn, cross_sim.shape)  # v * t
    return torch.Tensor(jaccard_sim).t().cuda()  # t * v


def textual_neighbor_mat(single_feat, cross_sim, neighbor_num=5):
    cross_sim = cross_sim.cpu().numpy()  # t * v
    single_sim = torch.matmul(single_feat, single_feat.t()).cpu().numpy()  # t * t

    cross_indices = np.argsort(-cross_sim, axis=1)
    single_indices = np.argsort(-single_sim, axis=1)
    cross_nn = cross_indices[:, :neighbor_num]  # t * n
    single_nn = single_indices[:, :neighbor_num]  # t * n

    jaccard_sim = jaccard_mat(cross_nn, single_nn, cross_sim.shape)  # t * v
    return torch.Tensor(jaccard_sim).cuda()


def get_unique(image_ids):
    keep_id = []
    tmp = 0
    for idx, image_id in enumerate(image_ids):
        if (image_id - tmp) > 0:
            keep_id.append(idx)
            tmp = image_id
    keep_id = torch.tensor(keep_id)
    return keep_id


def evaluation(
    dataset,
    predictions,
    output_folder,
    topk,
    save_data=True,
    rerank=True,
):
    logger = logging.getLogger("PersonSearch.inference")
    data_dir = os.path.join(output_folder, "inference_data.npz")

    if predictions is None:
        inference_data = np.load(data_dir)
        logger.info("Load inference data from {}".format(data_dir))
        image_pid = torch.tensor(inference_data["image_pid"])
        text_pid = torch.tensor(inference_data["text_pid"])
        similarity = torch.tensor(inference_data["similarity"])
        if rerank:
            rvn_mat = torch.tensor(inference_data["rvn_mat"])
            rtn_mat = torch.tensor(inference_data["rtn_mat"])
    else:
        image_ids, pids = [], []
        image_global, text_global = [], []

        # FIXME: need optimization
        for idx, prediction in predictions.items():
            image_id, pid = dataset.get_id_info(idx)
            image_ids.append(image_id)
            pids.append(pid)
            image_global.append(prediction[0])
            text_global.append(prediction[1])

        image_pid = torch.tensor(pids)
        text_pid = torch.tensor(pids)
        image_global = torch.stack(image_global, dim=0)
        text_global = torch.stack(text_global, dim=0)

        keep_id = get_unique(image_ids)
        image_global = image_global[keep_id]
        image_pid = image_pid[keep_id]

        image_global = F.normalize(image_global, p=2, dim=1)
        text_global = F.normalize(text_global, p=2, dim=1)

        similarity = torch.matmul(text_global, image_global.t())

        if rerank:
            rvn_mat = visual_neighbor_mat(image_global, similarity)
            rtn_mat = textual_neighbor_mat(text_global, similarity)

        if save_data:
            if not rerank:
                np.savez(
                    data_dir,
                    image_pid=image_pid.cpu().numpy(),
                    text_pid=text_pid.cpu().numpy(),
                    similarity=similarity.cpu().numpy(),
                )
            else:
                np.savez(
                    data_dir,
                    image_pid=image_pid.cpu().numpy(),
                    text_pid=text_pid.cpu().numpy(),
                    similarity=similarity.cpu().numpy(),
                    rvn_mat=rvn_mat.cpu().numpy(),
                    rtn_mat=rtn_mat.cpu().numpy(),
                )

    topk = torch.tensor(topk)

    if rerank:
        i2t_cmc, i2t_mAP, _ = rank(
            similarity.t(), image_pid, text_pid, topk, get_mAP=True
        )
        t2i_cmc, t2i_mAP, _ = rank(similarity, text_pid, image_pid, topk, get_mAP=True)
        re_i2t_cmc, re_i2t_mAP, _ = rank(
            rtn_mat.t() + similarity.t(), image_pid, text_pid, topk, get_mAP=True
        )
        re_t2i_cmc, re_t2i_mAP, _ = rank(
            rvn_mat + similarity, text_pid, image_pid, topk, get_mAP=True
        )
        cmc_results = torch.stack([topk, t2i_cmc, re_t2i_cmc, i2t_cmc, re_i2t_cmc])
        mAP_results = torch.stack(
            [torch.zeros_like(t2i_mAP), t2i_mAP, re_t2i_mAP, i2t_mAP, re_i2t_mAP]
        ).unsqueeze(-1)
        results = torch.cat([cmc_results, mAP_results], dim=1)
        results = results.t().cpu().numpy().tolist()
        results[-1][0] = "mAP"
        logger.info(
            "\n"
            + table_log(results, headers=["topk", "t2i", "re-t2i", "i2t", "re-i2t"])
        )
    else:
        t2i_cmc, _ = rank(similarity, text_pid, image_pid, topk, get_mAP=False)
        i2t_cmc, _ = rank(similarity.t(), image_pid, text_pid, topk, get_mAP=False)
        results = torch.stack((topk, t2i_cmc, i2t_cmc)).t().cpu().numpy()
        logger.info("\n" + table_log(results, headers=["topk", "t2i", "i2t"]))
    return t2i_cmc[0]
