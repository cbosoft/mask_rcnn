from typing import Tuple

import torch


class Augmentation:

    def __call__(self, image: torch.Tensor, target: dict) -> Tuple[torch.Tensor, dict]:
        return self.apply(image, target)

    def apply(self, image: torch.Tensor, target: dict) -> Tuple[torch.Tensor, dict]:
        raise NotImplementedError
