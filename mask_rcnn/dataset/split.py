from typing import List

import numpy as np
from torch.utils.data import Dataset, Subset

from ..config import CfgNode


def split_dataset(ds: Dataset, cfg: CfgNode) -> List[Dataset]:
    n = len(ds)
    indices = np.arange(n)
    np.random.shuffle(indices)
    pivot = int(n*0.8)
    train = Subset(ds, indices[:pivot])
    valid = Subset(ds, indices[pivot:])
    assert len(train)
    assert len(valid)
    return [train, valid]
