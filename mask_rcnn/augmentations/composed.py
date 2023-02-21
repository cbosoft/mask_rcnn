from typing import Tuple

import torch

from .aug_base import Augmentation


class ComposedAugmentations(Augmentation):

    def __init__(self, augs):
        self.augs = augs

    def apply(self, image: torch.Tensor, target: dict) -> Tuple[torch.Tensor, dict]:
        for aug in self.augs:
            image, target = aug(image, target)
        return image, target
