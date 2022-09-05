import torch


def to_float(v):
    if isinstance(v, torch.Tensor):
        return float(v.detach().cpu().item())
    return float(v)
