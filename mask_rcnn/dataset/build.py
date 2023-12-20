from torch.utils.data import DataLoader

from ..config import CfgNode

from .coco import COCODataset
from .images import ImagesDataset
from .split import split_dataset


def collate_fn(batch):
    return {k: [b[k] for b in batch] for k in batch[0].keys()}


def build_dataset(cfg: CfgNode, **kwargs):
    if cfg.data.is_classified_images:
        return ImagesDataset.from_config(cfg)
    else:
        return COCODataset.from_config(cfg, **kwargs)


def build_dataloaders(cfg: CfgNode):
    ds = build_dataset(cfg)
    dl_kws = dict(
        batch_size=cfg.training.batch_size,
        # shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )
    return [
        DataLoader(ds, shuffle=cfg.training.shuffle_every_epoch if i == 0 else False, **dl_kws)
        for i, ds in enumerate(split_dataset(ds, cfg))
    ]
