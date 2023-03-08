from typing import Tuple, List

import torch


class Augmentation:

    def __call__(self, images: List[torch.Tensor], targets: List[dict]) -> Tuple[List[torch.Tensor], List[dict]]:
        return self.apply(images, targets)

    def apply(self, images: List[torch.Tensor], targets: List[dict]) -> Tuple[List[torch.Tensor], List[dict]]:
        raise NotImplementedError
