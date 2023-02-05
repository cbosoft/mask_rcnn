from collections import defaultdict
from typing import Dict

import torch

from .base import MetricObject
from .iou import IntersectionOverUnion


class MetricsCollection(MetricObject):

    def __init__(self, *metrics: MetricObject):
        self.metrics = metrics
        self.score_thresh = 0.5
        self.iou_thresh = 0.5

    def batch_initialise(self):
        for metric in self.metrics:
            metric.batch_initialise()

    def batch_update(self, out, tgt):
        out = self.score_filter(out)
        out, tgt = self.align_data(out, tgt)
        for metric in self.metrics:
            metric.batch_update(out, tgt)

    def batch_finalise(self) -> Dict[str, float]:
        rv = dict()
        for metric in self.metrics:
            d = metric.batch_finalise()
            assert not set(d).intersection(rv)
            rv.update(d)
        return rv

    def score_filter(self, out: dict):
        scores = out['scores']
        filter_mask = scores > self.score_thresh
        return {k: v[filter_mask] for k, v in out.items()}

    def align_data(self, out, tgt):
        """
        Align two inference sets to facilitate direct comparison.

        Sometimes more or less objects are returned by the DL model. The objects that have been detected should be
        matched up (as best as possible) to target objects. This method performs this alignment by looking at objects
        detected and comparing them to targets using IoU. Objects detected with a high IoU with a target are matched up
        (indicating that the detected object specifically corresponds to that target).

        This works in basically the same way as the PyTorch `Matcher` utility:
        https://github.com/pytorch/vision/blob/master/torchvision/models/detection/_utils.py#L314

        :param out: Dict[str, torch.Tensor] of results of detection (i.e. output from model)
        :param tgt: Dict[str, torch.Tensor] of targets
        :return: Pair of dictionaries out, tgt with aligned values.
        """
        out_is_bigger = out['boxes'].shape[0] > tgt['boxes'].shape[0]

        bigger = (out if out_is_bigger else tgt)
        smaller = (tgt if out_is_bigger else out)
        bigger_boxes = bigger['boxes']
        smaller_boxes = smaller['boxes']
        bigger_n = bigger_boxes.shape[0]
        corresponding_indices = [-1]*bigger_n
        for i, bb in enumerate(bigger_boxes):
            max_iou = -1
            max_at = -1
            for j, sb in enumerate(smaller_boxes):
                iou = IntersectionOverUnion.box_iou(bb, sb)
                if iou > max_iou:
                    max_iou = iou
                    max_at = j
            if max_iou > self.iou_thresh:
                corresponding_indices[i] = max_at

        def aligned(s, b) -> torch.Tensor:
            rv = torch.zeros_like(b)
            for i, j in enumerate(corresponding_indices):
                if j >= 0:
                    rv[i] = s[j]
            return rv

        keys = {'boxes', 'masks', 'labels'}
        smaller_aligned = {
            k: aligned(smaller[k], bigger[k])
            for k in keys
        }
        tgt = smaller_aligned if out_is_bigger else tgt
        out = out if out_is_bigger else smaller_aligned
        return out, tgt
