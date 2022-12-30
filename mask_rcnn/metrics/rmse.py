from typing import Dict, List

import torch

from .base import MetricObject


class RMSE(MetricObject):
    """
    Square root of the mean of the square error
    """

    def __init__(self):
        self.squerrors = dict()

    def batch_initialise(self):
        self.squerrors = dict(boxes=[], labels=[], masks=[])

    def batch_update(self, out, tgt):
        for k in list(self.squerrors):
            self.squerrors[k].extend(self.se(out[k], tgt[k]))

    def batch_finalise(self) -> Dict[str, float]:
        rv = dict()
        for k, se in self.squerrors.items():
            rv[f'rmse.{k}'] = self.rm(se)
        return rv

    @staticmethod
    def rm(a: List[float]) -> float:
        return (sum(a)/len(a))**0.5

    @staticmethod
    def se(a: torch.Tensor, b: torch.Tensor) -> List[float]:
        n = min(a.shape[0], b.shape[0])
        a, b = a[:n], b[:n]
        se = (a - b)**2.
        return [float(f) for f in torch.flatten(se)]
