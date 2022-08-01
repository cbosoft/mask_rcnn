import torch.nn

from .config import CfgNode


def build_loss(cfg: CfgNode):
    return eval(cfg.training.loss, dict(nn=torch.nn))
