from yacs.config import CfgNode
import torch


def get_config() -> CfgNode:
    cfg = CfgNode()

    ####################################################################################################################
    cfg.data = CfgNode()

    # glob pattern specifying the json COCO-style files describing the data used in training
    cfg.data.pattern = None

    ####################################################################################################################
    cfg.model = CfgNode()
    cfg.model.n_classes = 1
    cfg.model.backbone = CfgNode()
    cfg.model.rpn_anchor_generator = CfgNode()
    cfg.model.roi_pooler = CfgNode()
    cfg.model.mask_roi_pooler = CfgNode()

    ####################################################################################################################
    cfg.training = CfgNode()

    # glob pattern specifying the json COCO-style files describing the data used in training
    cfg.training.n_epochs = 1_000
    cfg.training.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    cfg.training.loss = 'nn.MSELoss'
    cfg.training.batch_size = 10
    cfg.training.shuffle_every_epoch = True
    cfg.training.metrics = [
        'm.CosineSimilarity()',
        'm.RMSE()',
        'm.PearsonCorrcoef()',
        'm.R2Score()',
        'm.SpearmanCorrcoef()'
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
    cfg.training.sched.kind = 'OneCycle'  # ('OneCycle', 'Linear', 'Step')

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
