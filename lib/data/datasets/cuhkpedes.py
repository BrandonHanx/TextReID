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
        use_seg=True,
        use_att=True,
        max_length=100,
        max_attribute_length=25,
        transforms=None,
        crop_transforms=None,
        cap_transforms=None,
    ):
        self.root = root
        self.use_onehot = use_onehot
        self.use_seg = use_seg
        self.use_att = use_att
        self.max_length = max_length
        self.max_attribute_length = max_attribute_length
        self.transforms = transforms
        self.crop_transforms = crop_transforms
        self.cap_transforms = cap_transforms

        self.img_dir = os.path.join(self.root, "imgs")
        self.seg_dir = os.path.join(self.root, "segs")

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
        if self.use_seg:
            seg = Image.open(
                os.path.join(self.seg_dir, img_path.split(".")[0] + ".png")
            )
        else:
            seg = None

        if self.use_onehot:
            caption = data["onehot"]
            caption = torch.tensor(caption)
            caption = Caption([caption], max_length=self.max_length, padded=False)
        else:
            caption = data["sentence"]
            caption = Caption(caption)

        caption.add_field("img_path", img_path)

        if self.use_att:
            attribute = data["att_onehot"]
            if isinstance(attribute, dict):
                attribute_list = [torch.tensor(v) for k, v in attribute.items()]
            else:
                attribute_list = [torch.tensor(v) for v in attribute]
            attribute = Caption(
                attribute_list, max_length=self.max_attribute_length, padded=False
            )
            attribute.add_field("mask", attribute.length > 0)
            attribute.length[attribute.length < 1] = 1
            caption.add_field("attribute", attribute)

        label = data["id"]
        label = torch.tensor(label)
        caption.add_field("id", label)

        # gender = data["gender"]
        # gender = torch.tensor(gender)
        # caption.add_field("gender", gender)

        if self.transforms is not None:
            img, seg = self.transforms(img, seg)

        if self.crop_transforms is not None and self.use_seg:
            crops, mask = self.crop_transforms(img, seg)
            caption.add_field("crops", crops)  # value mask
            caption.add_field("mask", mask)  # existence mask

        if self.cap_transforms is not None:
            caption = self.cap_transforms(caption)

        return img, caption, index

    def __len__(self):
        return len(self.dataset)

    def get_id_info(self, index):
        image_id = self.dataset[index]["image_id"]
        pid = self.dataset[index]["id"]
        return image_id, pid
