import json
import os

import torch
from PIL import Image

from lib.utils.caption import Caption


class MSCOCODataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        ann_file,
        transforms=None,
    ):
        self.transforms = transforms
        self.img_dir = root

        print("loading annotations into memory...")
        dataset = json.load(open(ann_file, "r"))
        self.dataset = dataset

    def __getitem__(self, index):
        """
        Args:
              index(int): Index
        Returns:
              tuple: (images, labels, captions)
        """
        data = self.dataset[index]

        img_path = data["file_path"]
        img = Image.open(os.path.join(self.img_dir, img_path)).convert("RGB")

        caption = data["sentence"]
        caption = Caption(caption)

        label = data["id"]
        label = torch.tensor(label)
        caption.add_field("id", label)

        if self.transforms is not None:
            img, _ = self.transforms(img, None)

        return img, caption, index

    def __len__(self):
        return len(self.dataset)

    def get_id_info(self, index):
        image_id = self.dataset[index]["id"]
        pid = self.dataset[index]["id"]
        return image_id, pid
