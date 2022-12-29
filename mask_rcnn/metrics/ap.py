from typing import List, Dict

import torch
import numpy as np

from .base import MetricObject
from .iou import IntersectionOverUnion


class AveragePrecision(MetricObject):
    """
    Mean Average Precision
    https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173
    """

    def __init__(self):
        self.data_table = []
        self.pixel_thresh = 0.5

    def batch_initialise(self):
        self.data_table = []

    def batch_update(self, out: Dict[str, torch.Tensor], tgt: Dict[str, torch.Tensor]):
        self.data_table.extend(self.get_ap_data(out, tgt))

    def batch_finalise(self) -> Dict[str, float]:
        table = np.array(self.data_table)
        table = table[np.argsort(table[:, 2])]
        aps = []
        for iou_thresh in np.arange(0.5, 1.0, 0.05):
            table_t = table[table[:, 0] > iou_thresh]
            p = table_t[:, 1]
            r = table_t[:, 2]
            p_filtered = [p[i:].max() for i in range(len(r))]
            ap = np.trapz(p_filtered, r)
            aps.append(ap)
        return {
            'mask.mAP': float(np.mean(aps)),
            'mask.AP50': float(aps[0]),
            'mask.AP75': float(aps[5]),
            'mask.AP95': float(aps[-1]),
        }

    def get_ap_data(self, out: Dict[str, torch.Tensor], tgt: Dict[str, torch.Tensor]) -> List[list]:
        rv = []

        for omask, tmask in zip(out['masks'], tgt['masks']):
            iou, p, r = IntersectionOverUnion.mask_iou(omask, tmask, self.pixel_thresh, also_precision_recall=True)
            rv.append([iou, p, r])

        return rv
