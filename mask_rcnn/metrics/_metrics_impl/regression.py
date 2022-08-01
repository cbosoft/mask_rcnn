import torch

from .base import Metric


class RegressionMetric(Metric):
    def calculate_value(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class RMSE(RegressionMetric):
    def calculate_value(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = target - output
        square_error = error*error
        mean_square_error = torch.mean(square_error)
        root_mean_square_error = torch.sqrt(mean_square_error)
        return root_mean_square_error


class fractional_RMSE(RMSE):
    def calculate_value(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = target - output
        square_error = error*error/target/target
        mean_square_error = torch.mean(square_error)
        root_mean_square_error = torch.sqrt(mean_square_error)
        return root_mean_square_error


class CosineSimilarity(RegressionMetric):
    def calculate_value(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.dot(output, target) / torch.sqrt(torch.dot(output, output)*torch.dot(target, target))


class MeanAbsoluteError(RegressionMetric):
    def calculate_value(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(output - target))


class MeanAbsolutePercentageError(RegressionMetric):
    def calculate_value(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs((output - target)/target))*100.


class MeanAbsoluteFractionalError(RegressionMetric):
    def calculate_value(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs((output - target)/target))


class PearsonCorrcoef(RegressionMetric):
    def calculate_value(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        output_minus_mean = output - torch.mean(output)
        target_minus_mean = target - torch.mean(target)
        return torch.dot(output_minus_mean, target_minus_mean) / torch.sqrt(torch.dot(output_minus_mean, output_minus_mean)*torch.dot(target_minus_mean, target_minus_mean))


class R2Score(RegressionMetric):
    def calculate_value(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        y = output
        f = target
        y_minus_mean = y - torch.mean(y)
        y_minus_f = y - f
        ss_res = torch.dot(y_minus_f, y_minus_f)
        ss_tot = torch.dot(y_minus_mean, y_minus_mean)
        return 1. - ss_res/ss_tot


class SpearmanCorrcoef(RegressionMetric):

    def __init__(self):
        from torchmetrics.regression import SpearmanCorrCoef
        self.sc = SpearmanCorrCoef()

    def calculate_value(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.sc(output, target)
