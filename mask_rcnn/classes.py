from typing import Tuple

CLASSES = [
    'Elongated',
    'Regular',
    'Spherical',
    'Agglomerated',
    'Very Small',
    'User 1',
    'User 2',
    'User 3',
    'User 4',
    'User 5',
]


CLASS_COLOURS = [
    '#000000',  # 0. background
    '#1f77b4',  # 1. elongated
    '#ff7f0e',  # 2. regular
    '#2ca02c',  # 3. spherical
    '#d62728',  # 4. agglomerated
    '#9467bd',  # 5. very small
    '#8c564b',  # 6. user 1
    '#e377c2',  # 7. user 2
    '#7f7f7f',  # 8. user 3
    '#bcbd22',  # 9. user 4
    '#17becf',  # 19. user 5
]


def bgr_colour_for_class(i: int) -> Tuple[int, int, int]:
    hexstring = CLASS_COLOURS[i]
    r = hexstring[1:3]
    g = hexstring[3:5]
    b = hexstring[5:]
    r = int(r, 16)
    g = int(g, 16)
    b = int(b, 16)

    return b, g, r
