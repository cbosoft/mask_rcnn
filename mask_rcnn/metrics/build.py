import torch

from ..config import CfgNode
from . import _metrics_impl


def build_metrics(config: CfgNode) -> dict:
    device = torch.device(config.training.device)
    metrics = [
        eval(metric_src, dict(torchmetrics=_metrics_impl, m=_metrics_impl)).to(device)
        for metric_src in config.training.metrics]
    metrics = {m.__class__.__name__: m for m in metrics}
    return metrics
