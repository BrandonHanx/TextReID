# Text Based Person Search with Limited Data



[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/text-based-person-search-with-limited-data/nlp-based-person-retrival-on-cuhk-pedes)](https://paperswithcode.com/sota/nlp-based-person-retrival-on-cuhk-pedes?p=text-based-person-search-with-limited-data)

This is the codebase for our [BMVC 2021 paper](https://arxiv.org/abs/2110.10807).

Please bear with me refactoring this codebase after CVPR deadline :sweat_smile:

## Abstract
Text-based person search (TBPS) aims at retrieving a target person from an image gallery with a descriptive text query.
Solving such a fine-grained cross-modal retrieval task is challenging, which is further hampered by the lack of large-scale datasets.
In this paper, we present a framework with two novel components to handle the problems brought by limited data.
Firstly, to fully utilize the existing small-scale benchmarking datasets for more discriminative feature learning, we introduce a cross-modal momentum contrastive learning framework to enrich the training data for a given mini-batch. Secondly, we propose to transfer knowledge learned from existing coarse-grained large-scale datasets containing image-text pairs from drastically different problem domains to compensate for the lack of TBPS training data. A transfer learning method is designed so that useful information can be transferred despite the large domain gap.  Armed with these components, our method achieves new state of the art on the CUHK-PEDES dataset with significant improvements over the prior art in terms of Rank-1 and mAP.
