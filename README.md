# Text Based Person Search with Limited Data

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/text-based-person-search-with-limited-data/nlp-based-person-retrival-on-cuhk-pedes)](https://paperswithcode.com/sota/nlp-based-person-retrival-on-cuhk-pedes?p=text-based-person-search-with-limited-data)

This is the codebase for our [BMVC 2021 paper](https://arxiv.org/abs/2110.10807).

Slides and video for the online presentation are now available at [BMVC 2021 virtual conference website](https://www.bmvc2021-virtualconference.com/conference/papers/paper_0044.html).

## Updates
- (10/12/2021) Add download link of trained models.
- (06/12/2021) Code refactor for easy reproduce.
- (20/10/2021) Code released!

## Abstract
Text-based person search (TBPS) aims at retrieving a target person from an image gallery with a descriptive text query.
Solving such a fine-grained cross-modal retrieval task is challenging, which is further hampered by the lack of large-scale datasets.
In this paper, we present a framework with two novel components to handle the problems brought by limited data.
Firstly, to fully utilize the existing small-scale benchmarking datasets for more discriminative feature learning, we introduce a cross-modal momentum contrastive learning framework to enrich the training data for a given mini-batch. Secondly, we propose to transfer knowledge learned from existing coarse-grained large-scale datasets containing image-text pairs from drastically different problem domains to compensate for the lack of TBPS training data. A transfer learning method is designed so that useful information can be transferred despite the large domain gap.  Armed with these components, our method achieves new state of the art on the CUHK-PEDES dataset with significant improvements over the prior art in terms of Rank-1 and mAP.

## Results
![image](https://user-images.githubusercontent.com/37724292/144879635-86ab9c7b-0317-4b42-ac46-a37b06853d18.png)

## Installation
### Setup environment
```bash
conda create -n txtreid-env python=3.7
conda activate txtreid-env
git clone https://github.com/BrandonHanx/TextReID.git
cd TextReID
pip install -r requirements.txt
pre-commit install
```
### Get CUHK-PEDES dataset
- Request the images from [Dr. Shuang Li](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description).
- Download the pre-processed captions we provide from [Google Drive](https://drive.google.com/file/d/1V4d8OjFket5SaQmBVozFFeflNs6f9e1R/view?usp=sharing).
- Organize the dataset as following:
```bash
datasets
└── cuhkpedes
    ├── annotations
    │   ├── test.json
    │   ├── train.json
    │   └── val.json
    ├── clip_vocab_vit.npy
    └── imgs
        ├── cam_a
        ├── cam_b
        ├── CUHK01
        ├── CUHK03
        ├── Market
        ├── test_query
        └── train_query
```

### Download CLIP weights
```bash
mkdir pretrained/clip/
cd pretrained/clip
wget https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt
wget https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt
cd -

```

### Train
```bash
python train_net.py \
--config-file configs/cuhkpedes/moco_gru_cliprn50_ls_bs128_2048.yaml \
--use-tensorboard
```
### Inference
```bash
python test_net.py \
--config-file configs/cuhkpedes/moco_gru_cliprn50_ls_bs128_2048.yaml \
--checkpoint-file output/cuhkpedes/moco_gru_cliprn50_ls_bs128_2048/best.pth
```
You can download our trained models (with CLIP RN50 and RN101) from [Google Drive](https://drive.google.com/drive/folders/1MoceVsLiByg3Sg8_9yByGSvR3ru15hJL?usp=sharing).

## TODO
- [ ] Try larger pre-trained CLIP models.
- [ ] Fix the bug of multi-gpu runninng.
- [ ] Add dataloader for [ICFG-PEDES](https://github.com/zifyloo/SSAN).

## Citation
If you find this project useful for your research, please use the following BibTeX entry.
```
@inproceedings{han2021textreid,
  title={Text-Based Person Search with Limited Data},
  author={Han, Xiao and He, Sen and Zhang, Li and Xiang, Tao},
  booktitle={BMVC},
  year={2021}
}
```
