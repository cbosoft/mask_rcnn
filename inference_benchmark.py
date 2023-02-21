from time import process_time

import numpy as np
import torch
from matplotlib import pyplot as plt

from mask_rcnn.config import get_config, CfgNode
from mask_rcnn.model import build_model


def test_cfg(cfg: CfgNode, s: int, n=3):
    inp = torch.zeros((3, s, s))
    model = build_model(cfg)
    model.eval()

    times = []

    for _ in range(n):
        start = process_time()
        _ = model([inp])
        end = process_time()

        times.append(end - start)
    
    return np.mean(times)


if __name__ == '__main__':
    cfg = get_config()
    
    ns = []
    ss = []
    ts = []
    for s in range(100, 1000, 300):
        for n in [18, 34, 50, 101, 152]:
            cfg.model.backbone.resnet.n = n

            t = test_cfg(cfg, s)

            ns.append(n)
            ss.append(s)
            ts.append(t)

            print(f'mean inference time {t}s')
    
    plt.figure()
    plt.plot(ns, ts, 'o')
    plt.ylabel('Time to process image, $t$ [s]')
    plt.xlabel('ResNet Size, $n$')
    plt.show()