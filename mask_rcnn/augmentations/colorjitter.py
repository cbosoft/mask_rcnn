from typing import Tuple

import torch
from torchvision.transforms import ColorJitter as TV_ColorJitter

from .aug_base import Augmentation


class ColorJitter(Augmentation):

    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.0, hue=0.0):
        self._jitter = TV_ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def apply(self, image: torch.Tensor, target: dict) -> Tuple[torch.Tensor, dict]:
        transformed = self._jitter(image)
        return transformed, target
