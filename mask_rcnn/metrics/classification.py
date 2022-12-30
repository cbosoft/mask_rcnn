from typing import Dict, List

import torch

from .base import MetricObject


def ignore_zerodiv(inner):
    def outer(s):
        try:
            return inner(s)
        except ZeroDivisionError:
            return float('nan')
    return outer


class BinaryClassificationMetrics(MetricObject):
    """
    Standard binery classification metrics - (mask)

    https://en.wikipedia.org/wiki/Precision_and_recall#Definition_(classification_context)
    """

    METRICS = (
        'mask_sensitivity', 'mask_specificity',
        'mask_precision', 'mask_recall',
        'mask_fnr', 'mask_fpr',
        'mask_plr', 'mask_nlr',
        'mask_prevalence', 'mask_accuracy', 'mask_balancedaccuracy',
        'mask_f1', 'mask_mcc', 'mask_informedness', 'mask_dor',
    )

    def __init__(self):
        self.mask_fp, self.mask_fn, self.mask_tp, self.mask_tn = 0, 0, 0, 0
        self.pixel_thresh = 0.5

    def batch_initialise(self):
        self.mask_fp, self.mask_fn, self.mask_tp, self.mask_tn = 0, 0, 0, 0

    def batch_update(self, out, tgt):
        omask, tmask = out['masks'], tgt['masks']
        omask, tmask = omask > self.pixel_thresh, tmask > self.pixel_thresh
        self.mask_tp += int(torch.sum(tmask & omask).cpu())
        self.mask_fp += int(torch.sum(tmask & ~omask).cpu())
        self.mask_tn += int(torch.sum(~tmask & ~omask).cpu())
        self.mask_fn += int(torch.sum(~tmask & omask).cpu())

    @ignore_zerodiv
    def mask_sensitivity(self):
        """AKA true positive rate"""
        return self.mask_tp / (self.mask_tp + self.mask_fn)

    @ignore_zerodiv
    def mask_specificity(self):
        """AKA true negative rate"""
        return self.mask_tn / (self.mask_tn + self.mask_fp)

    @ignore_zerodiv
    def mask_precision(self):
        """AKA positive predictive value"""
        return self.mask_tp / (self.mask_tp + self.mask_fp)

    @ignore_zerodiv
    def mask_recall(self):
        return self.mask_tp / (self.mask_tp + self.mask_fn)

    @ignore_zerodiv
    def mask_fnr(self):
        """false negative rate"""
        return 1 - self.mask_specificity()

    @ignore_zerodiv
    def mask_fpr(self):
        """false positive rate"""
        return 1 - self.mask_sensitivity()

    @ignore_zerodiv
    def mask_plr(self):
        """positive likelihood ratio"""
        return self.mask_sensitivity() / self.mask_fpr()

    @ignore_zerodiv
    def mask_nlr(self):
        """negative likelihood ratio"""
        return self.mask_fnr() / self.mask_specificity()

    @ignore_zerodiv
    def mask_prevalence(self):
        return (self.mask_tp + self.mask_fp) / (self.mask_tp + self.mask_tn + self.mask_fp + self.mask_fn)

    @ignore_zerodiv
    def mask_accuracy(self):
        return (self.mask_tp + self.mask_tn) / (self.mask_tp + self.mask_tn + self.mask_fp + self.mask_fn)

    def mask_balancedaccuracy(self):
        return (self.mask_sensitivity() + self.mask_specificity())*0.5

    @ignore_zerodiv
    def mask_f1(self):
        return 2.0*self.mask_tp/(2*self.mask_tp + self.mask_fp + self.mask_fn)

    @ignore_zerodiv
    def mask_mcc(self):
        """matthews correlation coefficient"""
        tp, fp, tn, fn = self.mask_fp, self.mask_fp, self.mask_tn, self.mask_fn
        return (tp*tn - fp*fn)/(((tp + fp)*(tp+fn)*(tn+fp)*(tn+fn))**0.5)

    def mask_informedness(self):
        """bookmakers informedness"""
        return self.mask_sensitivity() + self.mask_sensitivity() - 1

    @ignore_zerodiv
    def mask_dor(self):
        """diagnostic odds ratio"""
        return self.mask_plr() / self.mask_nlr()

    def batch_finalise(self) -> Dict[str, float]:
        return {metric_name.replace('_', '.'): getattr(self, metric_name)() for metric_name in self.METRICS}
