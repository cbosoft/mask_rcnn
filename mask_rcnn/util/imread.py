from typing import Tuple

import torch
import numpy as np
import cv2


def imread(fn: str, size: Tuple[int, int] = None) -> torch.Tensor:
    im: np.ndarray = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise IOError(f'Error reading image "{fn}"!')
    if size is not None:
        im = cv2.resize(im, size)
    return torch.tensor(im)
