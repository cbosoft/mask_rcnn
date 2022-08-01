from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import cv2


def imread(fn: str, size: Tuple[int, int] = None) -> torch.Tensor:
    im: np.ndarray = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise IOError(f'Error reading image "{fn}"!')
    if size is not None:
        im = cv2.resize(im, size)
    return torch.tensor(im)


def today():
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def ensure_dir(dirname: str) -> str:
    Path(dirname).mkdir(parents=True, exist_ok=True)
    return dirname


def gaussian_pdf(x, mu, std):
    pdf = 1./std/np.sqrt(2.*np.pi)*np.exp(-0.5*np.square((x - mu)/std))
    # not always normalised properly? binning issue?
    pdf /= np.trapz(pdf, x)
    return pdf


def geometric_mean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a*b)**.5


def geometric_midp(v: np.ndarray) -> np.ndarray:
    return geometric_mean(v[1:], v[:-1])


def to_float(v):
    if isinstance(v, torch.Tensor):
        return float(v.detach().cpu().item())
    return float(v)


def onehot(v, n):
    rv = torch.zeros(n)
    rv[v] = 1
    return rv

