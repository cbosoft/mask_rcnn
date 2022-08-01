import torch.optim.lr_scheduler as lr_scheduler

from .config import CfgNode


def build_sched(config: CfgNode, batches_per_epoch: int):
    kind = config.training.sched.kind
    if kind == 'OneCycle':
        sched_t = lr_scheduler.OneCycleLR
        sched_kws = dict(config.training.sched.onecycle)
        sched_kws['epochs'] = config.training.n_epochs
        sched_kws['steps_per_epoch'] = batches_per_epoch
    elif kind == 'Linear':
        sched_t = lr_scheduler.LinearLR
        sched_kws = dict(config.training.sched.linear)
        if sched_kws['total_iters'] is None:
            sched_kws['total_iters'] = config.training.n_epochs*batches_per_epoch
    elif kind == 'Step':
        sched_t = lr_scheduler.StepLR
        sched_kws = dict(config.training.sched.step)
    elif kind == 'Exponential':
        sched_t = lr_scheduler.ExponentialLR
        sched_kws = dict(config.training.sched.exponential)
    else:
        raise ValueError(f'Didn\'t understand given scheduler: "{kind}"')

    return sched_t, sched_kws
