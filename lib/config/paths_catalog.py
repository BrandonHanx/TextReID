# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Centralized catalog of paths."""

import os


class DatasetCatalog:
    DATA_DIR = "datasets"
    DATASETS = {
        "cuhkpedes_train": {
            "img_dir": "cuhkpedes",
            "ann_file": "cuhkpedes/annotations/train.json",
        },
        "cuhkpedes_val": {
            "img_dir": "cuhkpedes",
            "ann_file": "cuhkpedes/annotations/val.json",
        },
        "cuhkpedes_test": {
            "img_dir": "cuhkpedes",
            "ann_file": "cuhkpedes/annotations/test.json",
        },
        "mscoco_train": {
            "img_dir": "mscoco",
            "ann_file": "mscoco/annotations/train.json",
        },
        "mscoco_val": {
            "img_dir": "mscoco",
            "ann_file": "mscoco/annotations/val.json",
        },
        "mscoco_rv": {
            "img_dir": "mscoco",
            "ann_file": "mscoco/annotations/restval.json",
        },
        "mscoco_test": {
            "img_dir": "mscoco",
            "ann_file": "mscoco/annotations/test.json",
        },
        "mscoco_val_1k": {
            "img_dir": "mscoco",
            "ann_file": "mscoco/annotations/val_1k.json",
        },
        "mscoco_test_1k": {
            "img_dir": "mscoco",
            "ann_file": "mscoco/annotations/test_1k.json",
        },
    }

    @staticmethod
    def get(name):
        if "cuhkpedes" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="CUHKPEDESDataset",
                args=args,
            )
        elif "mscoco" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="MSCOCODataset",
                args=args,
            )
        raise RuntimeError("Dataset not available: {}".format(name))
