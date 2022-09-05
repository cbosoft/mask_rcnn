from typing import Tuple
import numpy as np


def midp(pt1, pt2) -> Tuple[float, float]:
    return np.mean([pt1, pt2], axis=0)


def dist(pt1, pt2) -> float:
    return np.sum((pt2 - pt1) ** 2) ** 0.5


def get_axes(box) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Get two axis of box (e.g. horiz. and vert.).

    Returns list of tuples of two points defining each axis.
    """
    tl, tr, br, bl = box
    t = midp(tl, tr)
    b = midp(bl, br)
    l = midp(tl, bl)
    r = midp(tr, br)
    return (t, b), (l, r)


def size_of_box(box) -> Tuple[float, float]:
    """
    Get size of rectange with corners defined by $box.

    Return (width, length).
    """
    axA, axB = get_axes(box)
    size = dist(*axA), dist(*axB)
    size = tuple(sorted(size))
    return size
