import math
import random

import torch
from PIL import Image
from torchvision.transforms import functional as F


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, seg=None):
        for t in self.transforms:
            image, seg = t(image, seg)
        return image, seg

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize:
    def __init__(self, size, ratio, interpolation=Image.BILINEAR):
        self.img_size = size
        self.seg_size = (int(size[0] * ratio), int(size[1] * ratio))
        self.interpolation = interpolation

    def __call__(self, image, seg=None):
        image = F.resize(image, self.img_size, self.interpolation)
        seg = F.resize(seg, self.seg_size, Image.NEAREST) if seg else None
        return image, seg


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, seg=None):
        if random.random() < self.prob:
            image = F.hflip(image)
            seg = F.hflip(seg) if seg else None
        return image, seg


class Pad:
    def __init__(self, padding, fill=0, padding_mode="constant"):
        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, image, seg=None):
        image = F.pad(image, self.padding, self.fill, self.padding_mode)
        seg = F.pad(seg, self.padding, self.fill, self.padding_mode) if seg else None
        return image, seg


class RandomCrop:
    def __init__(self, size):
        self.size = size

    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, image, seg=None):
        i, j, h, w = self.get_params(image, self.size)
        image = F.crop(image, i, j, h, w)
        seg = F.crop(seg, i, j, h, w) if seg else None
        return image, seg


class ToTensor:
    def __call__(self, image, seg=None):
        image = F.to_tensor(image)
        seg = F.to_tensor(seg) if seg else None
        return image, seg


class Normalize:
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, seg=None):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        seg = 255 * seg[0] if seg is not None else None
        return image, seg


class RandomErasing:
    """Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(
        self, prob=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]
    ):
        self.prob = prob
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, image, seg=None):
        if random.uniform(0, 1) >= self.prob:
            return image, seg

        for _ in range(100):
            area = image.size(1) * image.size(2)

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if h < image.size(1) and w < image.size(2):
                x1 = random.randint(0, image.size(1) - h)
                y1 = random.randint(0, image.size(2) - w)

                image[0, x1 : x1 + h, y1 : y1 + w] = self.mean[0]
                image[1, x1 : x1 + h, y1 : y1 + w] = self.mean[1]
                image[2, x1 : x1 + h, y1 : y1 + w] = self.mean[2]
                return image, seg
        return image, seg


class Split:
    def __init__(self, num_parts, use_binary=False):
        self.num_parts = num_parts
        self.use_binary = use_binary

    def __call__(self, image, seg):
        masks = []
        part_masks = []
        label_set = list(set(seg.numpy().flatten()))
        for label in range(1, self.num_parts + 1):
            if label in label_set:
                label = 1 if self.use_binary else label
                mask = (seg == label) * label
                part_masks.append(True)
            else:
                mask = torch.zeros_like(seg, dtype=torch.uint8)
                part_masks.append(False)
            masks.append(mask)
        return torch.stack(masks), torch.tensor(part_masks)
