from typing import Dict

import torch

from .base import MetricObject


class IntersectionOverUnion(MetricObject):
    """
    Intersection over Union
    https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173
    """

    def __init__(self):
        self.iou = dict()
        self.pixel_thresh = 0.5

    def batch_initialise(self):
        self.iou = dict(box=[], mask=[])

    def batch_update(self, out, tgt):
        for ob, tb in zip(out['boxes'], tgt['boxes']):
            self.iou['box'].append(self.box_iou(ob, tb))

        for ob, tb in zip(out['masks'], tgt['masks']):
            self.iou['mask'].append(self.mask_iou(ob, tb, self.pixel_thresh))

    def batch_finalise(self) -> Dict[str, float]:
        rv = dict()
        for k, iou in self.iou.items():
            rv[f'iou.{k}'] = sum(iou)/len(iou)
        return rv

    @staticmethod
    def mask_iou(a: torch.Tensor, b: torch.Tensor, pixel_thresh: float) -> float:
        a = a > pixel_thresh
        b = b > pixel_thresh
        intersection = torch.sum(a & b)
        union = torch.sum(a | b)
        return float(intersection / union)

    @staticmethod
    def boxes_do_interset(a, b):
        left, right = (a, b) if (a[0] < b[0]) else (b, a)
        top, bottom = (b, a) if (a[1] < b[1]) else (a, b)
        return (left[2] > right[0]) and (bottom[3] > top[1])

    @classmethod
    def box_iou(cls, a: torch.Tensor, b: torch.Tensor) -> float:
        """
        from:
        https://stackoverflow.com/questions/28723670/intersection-over-union-between-two-detections

        :param a: bbox (x1, y1, x2, y2)
        :param b: another bbox (x1, y1, x2, y2)
        :return: IoU of the two bboxes
        """

        if not cls.boxes_do_interset(a, b):
            return 0.0

        # determine the (x, y)-coordinates of the intersection rectangle
        xA = float(max(a[0], b[0]))
        yA = float(max(a[1], b[1]))
        xB = float(min(a[2], b[2]))
        yB = float(min(a[3], b[3]))

        # compute the area of intersection rectangle
        interArea = (xB - xA) * (yB - yA)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = abs((a[2] - a[0]) * (a[3] - a[1]))
        boxBArea = abs((b[2] - b[0]) * (b[3] - b[1]))

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou
