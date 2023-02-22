import random
from typing import Tuple

import torch

from .aug_base import Augmentation


class RandomFlip(Augmentation):

    """
    randomly flip image horixontally or vertically
    """

    def __init__(self, hflips=True, vflips=True, rots=True):
        self.hflips = hflips
        self.vflips = vflips
        self.rots = rots

    @staticmethod
    def hflip(images: torch.Tensor, targets: dict):
        for image, target in zip(images, targets):
            image[:] = torch.flip(image, [2])
            w = image.shape[2]
            for box in target['boxes']:
                x1, x2 = box[0].item(), box[2].item()
                box[0] = w - x2
                box[2] = w - x1
            for mask in target['masks']:
                mask[:] = torch.flip(mask, [1])

    @staticmethod
    def vflip(images: torch.Tensor, targets: dict):
        for image, target in zip(images, targets):
            h = image.shape[1]
            image[:] = torch.flip(image, [1])
            for box in target['boxes']:
                y1, y2 = box[1].item(), box[3].item()
                box[1] = h - y2
                box[3] = h - y1
            for mask in target['masks']:
                mask[:] = torch.flip(mask, [0])


    def apply(self, image: torch.Tensor, target: dict) -> Tuple[torch.Tensor, dict]:
        if self.hflips and random.randint(0, 1):
            self.hflip(image, target)

        if self.vflips and random.randint(0, 1):
            self.vflip(image, target)

        return image, target
