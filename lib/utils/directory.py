import os

import numpy as np


def makedir(root):
    if not os.path.exists(root):
        os.makedirs(root)


def load_vocab_dict(root, use_onehot):
    if use_onehot == "bert_c4":
        vocab_dict = np.load(
            os.path.join(root, "./datasets/cuhkpedes/bert_vocab_c4.npy")
        )
    elif use_onehot == "bert_l2":
        vocab_dict = np.load(
            os.path.join(root, "./datasets/cuhkpedes/bert_vocab_l2.npy")
        )
    elif use_onehot == "clip_vit":
        vocab_dict = np.load(
            os.path.join(root, "./datasets/cuhkpedes/clip_vocab_vit.npy")
        )
    elif use_onehot == "clip_rn50x4":
        vocab_dict = np.load(
            os.path.join(root, "./datasets/cuhkpedes/clip_vocab_rn50x4.npy")
        )
    else:
        NotImplementedError
    return vocab_dict
