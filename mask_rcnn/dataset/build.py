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
    dl_kws = dict(
        batch_size=cfg.training.batch_size,
        drop_last=True,
    )
    return [
        DataLoader(ds, shuffle=cfg.training.shuffle_every_epoch if i == 0 else False, **dl_kws)
        for i, ds in enumerate(split_dataset(ds, cfg))
    ]
