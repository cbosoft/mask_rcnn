from torch.utils.data import DataLoader

from ..config import CfgNode

from .coco import COCODataset
from .split import split_dataset


def collate_fn(*args):
    print('COLLATE ARGS', args)
    return args[0]


def build_dataset(cfg: CfgNode):
    return COCODataset.from_config(cfg)


def build_dataloaders(cfg: CfgNode):
    ds = build_dataset(cfg)
    return [DataLoader(ds) for ds in split_dataset(ds, cfg)]
