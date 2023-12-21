import random
import numpy as np
import torch


def set_seed(v):
    random.seed(v)
    np.random.seed(v)
    torch.manual_seed(v)
    print(f'Using seed {v}')

