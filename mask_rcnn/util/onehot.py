import torch


def onehot(v, n):
    rv = torch.zeros(n)
    rv[v] = 1
    return rv
