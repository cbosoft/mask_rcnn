from time import process_time

import git
import numpy as np
import torch
from matplotlib import pyplot as plt

from mask_rcnn.config import get_config, CfgNode
from mask_rcnn.model import build_model


def test_cfg(cfg: CfgNode, device: str, s: int, n=3):
    inp = torch.zeros((3, s, s))
    model = build_model(cfg, quiet=True)
    model.eval()

    model = model.to(device)
    inp = inp.to(device)

    times = []

    for _ in range(n):
        start = process_time()
        _ = model([inp])
        end = process_time()

        times.append(end - start)
    
    return np.mean(times)


if __name__ == '__main__':
    cfg = get_config()

    repo = git.Repo('.')
    version_name = repo.commit().hexsha[:6] + ('+' if repo.is_dirty() else '')
    print(version_name)
    
    ns = []
    ss = []
    ts = []
    gs = []
    for g in [0, 1]:
        for s in [500, 1000, 2000, 4000]:
            for n in [18, 34, 50, 101, 152]:
                cfg.model.backbone.resnet.n = n

                device = 'cuda' if g else 'cpu'

                t = test_cfg(cfg, device, s)

                ns.append(n)
                ss.append(s)
                ts.append(t)

                with open('benchmark_results.csv', 'a') as f:
                    f.write(f'{version_name},{n},{s},{device},{t}\n')

                print(f'mean inference time {t}s')
    
    plt.figure()
    plt.plot(ns, ts, 'o')
    plt.ylabel('Time to process image, $t$ [s]')
    plt.xlabel('ResNet Size, $n$')
    plt.show()
