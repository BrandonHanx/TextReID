import json
import os

import torch
from PIL import Image

from lib.utils.caption import Caption


class CUHKPEDESDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        ann_file,
        use_onehot=True,
        max_length=100,
        transforms=None,
    ):
        self.root = root
        self.use_onehot = use_onehot
        self.max_length = max_length
        self.transforms = transforms

        self.img_dir = os.path.join(self.root, "imgs")

        print("loading annotations into memory...")
        dataset = json.load(open(ann_file, "r"))
        self.dataset = dataset["annotations"]

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

        if self.use_onehot:
            caption = data["onehot"]
            caption = torch.tensor(caption)
            caption = Caption([caption], max_length=self.max_length, padded=False)
        else:
            caption = data["sentence"]
            caption = Caption(caption)

        caption.add_field("img_path", img_path)

        label = data["id"]
        label = torch.tensor(label)
        caption.add_field("id", label)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, caption, index

    def __len__(self):
        return len(self.dataset)

    def get_id_info(self, index):
        image_id = self.dataset[index]["image_id"]
        pid = self.dataset[index]["id"]
        return image_id, pid
