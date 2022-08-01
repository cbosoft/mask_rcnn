import torch

from .base import Metric


class ClassificationMetric(Metric):

    @staticmethod
    def get_tp_etc(output: torch.Tensor, target: torch.Tensor):
        true = output == target
        false = output != target
        tp = torch.sum(true & output)
        tn = torch.sum(true & ~output)
        fp = torch.sum(false & output)
        fn = torch.sum(false & ~output)
        return tp, tn, fp, fn

    def calculate_value(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Accuracy(ClassificationMetric):

    def calculate_value(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        tp, tn, fp, fn = self.get_tp_etc(output, target)
        return (tp + tn) / (tp + fp + tn + fn)


class Prevalence(ClassificationMetric):

    def calculate_value(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        tp, tn, fp, fn = self.get_tp_etc(output, target)
        return (tp + fp) / (tp + fp + tn + fn)


class Precision(ClassificationMetric):

    def calculate_value(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        tp, tn, fp, fn = self.get_tp_etc(output, target)
        return tp / (tp + fp)


class Recall(ClassificationMetric):

    def calculate_value(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        tp, tn, fp, fn = self.get_tp_etc(output, target)
        return tp / (tp + fn)


class Fhalf(ClassificationMetric):

    def calculate_value(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        tp, tn, fp, fn = self.get_tp_etc(output, target)
        precision = tp / (tp + fn)
        recall = tp / (tp + fp)
        return 1.25*precision*recall/(0.25*precision + recall)


class F1(ClassificationMetric):

    def calculate_value(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        tp, tn, fp, fn = self.get_tp_etc(output, target)
        precision = tp / (tp + fn)
        recall = tp / (tp + fp)
        return 2.*precision*recall/(precision + recall)


class F2(ClassificationMetric):

    def calculate_value(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        tp, tn, fp, fn = self.get_tp_etc(output, target)
        precision = tp / (tp + fn)
        recall = tp / (tp + fp)
        return 5.*precision*recall/(4.*precision + recall)
