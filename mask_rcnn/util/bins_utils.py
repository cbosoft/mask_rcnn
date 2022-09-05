import numpy as np


def geometric_mean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a*b)**.5


def geometric_midp(v: np.ndarray) -> np.ndarray:
    return geometric_mean(v[1:], v[:-1])
