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
    }

    @staticmethod
    def get(root, name):
        if "cuhkpedes" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(root, data_dir, attrs["img_dir"]),
                ann_file=os.path.join(root, data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="CUHKPEDESDataset",
                args=args,
            )
        raise RuntimeError("Dataset not available: {}".format(name))
