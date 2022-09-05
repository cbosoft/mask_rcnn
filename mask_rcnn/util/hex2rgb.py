from typing import Tuple


def hex2rgb(h: str, order='rgb') -> Tuple[float, float, float]:
    h = h.lstrip('#')
    r = int(h[:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:], 16)

    if order == 'rgb':
        return r, g, b
    elif order == 'bgr':
        return b, g, r
    else:
        raise NotImplementedError
