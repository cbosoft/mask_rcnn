import torch.optim as optim

from .config import CfgNode


def build_optim(config: CfgNode):
    kind = config.training.opt.kind
    if kind == 'Adam':
        opt_t = optim.Adam
        opt_kws = dict(config.training.opt.adam)
    elif kind == 'SGD':
        opt_t = optim.SGD
        opt_kws = dict(config.training.opt.sgd)
    else:
        raise ValueError(f'Didn\'t understand given optimiser: "{kind}"')

    return opt_t, opt_kws
