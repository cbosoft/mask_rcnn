from ..config import CfgNode
from .aug_base import Augmentation
from .composed import ComposedAugmentations
from .random_flip import RandomFlip
from .colorjitter import ColorJitter


AUGS = dict(
    RandomFlip=RandomFlip,
    ColorJitter=ColorJitter,
)


def build_augmentations(cfg: CfgNode) -> Augmentation:
    return ComposedAugmentations([
        eval(a_src, dict(**AUGS))
        for a_src in cfg.data.augmentations
    ])
