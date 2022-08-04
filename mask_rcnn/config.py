import os
from datetime import datetime

from yacs.config import CfgNode
import torch


def get_config() -> CfgNode:
    cfg = CfgNode()

    # What do we want to do in the experiment?
    # Can only be 'train'.
    # Future versions will include 'xval' (cross validation), and 'inference'.
    cfg.action = 'train'

    # set to True to output information useful for debug
    cfg.debug_mode = False

    # pattern to use to create the output directory, where training results are stored.
    cfg.output_dir = 'training_results/%Y-%m-%d_%H-%M-%S'

    ####################################################################################################################
    cfg.data = CfgNode()

    # glob pattern or patterns specifying the json COCO-style files describing the data used in training
    cfg.data.pattern = None

    ####################################################################################################################
    cfg.model = CfgNode()

    # load a previous state from file or database, or leave as is.
    # If None, no state is loaded
    # If a string, the string must be a path pointing to a state file (*.pth)
    # Otherwise, must be a tuple of (exp_id, epoch) used to find a model state file in the database.
    cfg.model.state = None

    cfg.model.n_classes = 1
    cfg.model.backbone = CfgNode()
    cfg.model.backbone.kind = 'resnet'
    cfg.model.backbone.pretrained = True

    # Optional string (python code) defining the layers to be returned by the backbone. Should eval to list of int.
    # e.g. 'list(range(1, 5))' -> [1, 2, 3, 4]
    cfg.model.backbone.returned_layers = None

    # How many layers should be trainable? The rest are frozen.
    cfg.model.backbone.trainable_layers = 2

    # ResNet specific backbone settings
    cfg.model.backbone.resnet = CfgNode()
    cfg.model.backbone.resnet.n = 18

    cfg.model.rpn_anchor_generator = CfgNode()
    cfg.model.roi_pooler = CfgNode()
    cfg.model.mask_roi_pooler = CfgNode()

    ####################################################################################################################
    cfg.training = CfgNode()
    cfg.training.checkpoint_every = 100
    cfg.training.visualise_every = 5

    # glob pattern specifying the json COCO-style files describing the data used in training
    cfg.training.n_epochs = 1_000
    cfg.training.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    cfg.training.loss = 'nn.MSELoss'
    cfg.training.batch_size = 10
    cfg.training.shuffle_every_epoch = True
    cfg.training.metrics = [
        # 'm.CosineSimilarity()',
        # 'm.RMSE()',
        # 'm.PearsonCorrcoef()',
        # 'm.R2Score()',
        # 'm.SpearmanCorrcoef()'
    ]

    cfg.training.opt = CfgNode()
    cfg.training.opt.kind = 'Adam'  # ('Adam', 'SGD')

    # Adam optimiser. See pytorch docs for param info:
    # https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam
    cfg.training.opt.adam = CfgNode()
    cfg.training.opt.adam.lr = 1e-3
    cfg.training.opt.adam.betas = (0.9, 0.999)
    cfg.training.opt.adam.weight_decay = 0.0
    cfg.training.opt.adam.amsgrad = False

    # SGD optimiser. See pytorch docs for param info:
    # https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD
    cfg.training.opt.sgd = CfgNode()
    cfg.training.opt.sgd.lr = 1e-3
    cfg.training.opt.sgd.momentum = 0.0
    cfg.training.opt.sgd.dampening = 0.0
    cfg.training.opt.sgd.weight_decay = 0.0
    cfg.training.opt.sgd.nesterov = False
    cfg.training.opt.sgd.maximize = False

    cfg.training.sched = CfgNode()
    cfg.training.sched.kind = 'OneCycle'  # ('OneCycle', 'Linear', 'Step', 'None')

    # OneCycle LR scheduler. See pytorch docs for param info:
    # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html#torch.optim.lr_scheduler.OneCycleLR
    cfg.training.sched.onecycle = CfgNode()
    cfg.training.sched.onecycle.max_lr = 0.01
    cfg.training.sched.onecycle.pct_start = 0.3
    cfg.training.sched.onecycle.anneal_strategy = 'cos'
    cfg.training.sched.onecycle.cycle_momentum = True
    cfg.training.sched.onecycle.base_momentum = 0.85
    cfg.training.sched.onecycle.max_momentum = 0.95
    cfg.training.sched.onecycle.div_factor = 25.0
    cfg.training.sched.onecycle.final_div_factor = 10000.0
    cfg.training.sched.onecycle.three_phase = False

    # Linear LR scheduler. See pytorch docs for more info:
    # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LinearLR.html#torch.optim.lr_scheduler.LinearLR
    cfg.training.sched.linear = CfgNode()
    cfg.training.sched.linear.start_factor = 1. / 3.
    cfg.training.sched.linear.end_factor = 1.0
    cfg.training.sched.linear.total_iters = None  # set to value to only decay for that number of steps (batches)

    # Step LR scheduler. See pytorch docs for more info:
    # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html#torch.optim.lr_scheduler.StepLR
    cfg.training.sched.step = CfgNode()
    cfg.training.sched.step.step_size = 100
    cfg.training.sched.step.gamma = 0.1

    # Step LR scheduler. See pytorch docs for more info:
    # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ExponentialLR.html
    cfg.training.sched.exponential = CfgNode()
    cfg.training.sched.exponential.gamma = 0.1

    return cfg


def finalise(cfg: CfgNode):
    # data spec
    assert cfg.data.pattern is not None, 'cfg.data.pattern must be specified and must be a string or a list of strings.'
    if isinstance(cfg.data.pattern, str):
        cfg.data.pattern = [cfg.data.pattern]

    # output dir
    cfg.output_dir = datetime.now().strftime(cfg.output_dir)
    os.makedirs(cfg.output_dir, exist_ok=True)

    # model state
    if cfg.model.state is not None:
        assert isinstance(cfg.model.state, (str, tuple)), f'cfg.model.state must be string or 2-tuple: (expid: str, epoch: int) (got {type(cfg.model.state)}).'
        if isinstance(cfg.model.state, tuple):
            assert len(cfg.model.state) == 2, f'cfg.model.state tuple must be 2-tuple: (expid: str, epoch: int) (got {len(cfg.model.state)})'
            expid, epoch = cfg.model.state
            assert isinstance(expid, str), f'cfg.model.state item 0 should be string experiment id (got {expid})'
            assert isinstance(epoch, int), f'cfg.model.state item 1 should be int epoch # (got {epoch})'

    # freeze config, making it immutable.
    cfg.freeze()
