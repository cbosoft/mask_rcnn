from typing import Dict
import torch

from ..config import CfgNode
from .base import MetricObject
from .metrics_collection import MetricsCollection
from .rmse import RMSE
from .iou import IntersectionOverUnion
from .classification import BinaryClassificationMetrics
from .ap import AveragePrecision


METRICS = dict(
    RMSE=RMSE,
    IoU=IntersectionOverUnion,
    BinaryClassificationMetrics=BinaryClassificationMetrics,
    mAP=AveragePrecision
)


def metric_from_source(metric_src: str) -> MetricObject:
    o = eval(metric_src, METRICS)
    if not isinstance(o, MetricObject):
        o = o()
    return o


def build_metrics(config: CfgNode) -> MetricsCollection:
    metrics = [
        metric_from_source(metric_src)
        for metric_src in config.training.metrics
    ]
    return MetricsCollection(*metrics)
