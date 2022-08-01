import torch


class Metric:

    def __call__(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.calculate_value(output, target)

    def calculate_value(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def to(self, *_, **__):
        return self
