import os
from datetime import datetime

from syt import CfgNode

import torch


def get_config() -> CfgNode:
    cfg = CfgNode()

    # What do we want to do in the experiment?
    # Can be 'train', 'xval'
    # Future versions will include 'inference'.
    cfg.action = 'train'

    # set to True to output information useful for debug
    cfg.debug_mode = False

    # pattern to use to create the output directory, where training results are stored.
    cfg.output_dir = 'training_results/%Y-%m-%d_%H-%M-%S'
    cfg.group = 'Mask R-CNN exp;data={data};arch={arch};tl={tl};n={n};e={e};sched={sched};opt={opt};augs={augs};'
    cfg.tag = None

    ####################################################################################################################
    cfg.data = CfgNode()

    # glob pattern or patterns specifying the json COCO-style files describing the data used in training
    cfg.data.pattern = None

    # For contrastive learning, data just needs to be a bunch of images and some indication of their class.
    # Specify this dataset type by setting this to True.
    cfg.data.is_classified_images = False

    # Function that runs on the image filenames to get class. By default it returns the name of the parent directory of the images.
    # Potentially useful to classify by top two dirs or by metadata encoded in image file.
    # Function that takes in a string and returns a string, AKA: Callable[[str], str]
    cfg.data.classifier_func = 'lambda fn: os.path.basename(os.path.dirname(fn))'

    cfg.data.max_size = 1024
    cfg.data.min_size = 320

    cfg.data.frac_test = 0.2

    # TODO
    cfg.data.max_number_images = -1

    cfg.data.augmentations = [
        # 'RandomFlip()'
    ]

    ####################################################################################################################
    cfg.model = CfgNode()

    # load a previous state from file or database, or leave as is.
    # If None, no state is loaded
    # If a string, the string must be a path pointing to a state file (*.pth)
    # Otherwise, must be a tuple of (exp_id, epoch) used to find a model state file in the database.
    cfg.model.state = None

    cfg.model.n_classes = 11
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
    # affects GPU usage
    cfg.model.rpn_anchor_generator.sizes = [32, 64, 128, 256, 512]
    cfg.model.rpn_anchor_generator.aspect_ratios = [0.5, 1.0, 2.0]
    cfg.model.rpn_batch_size_per_image = 256
    
    cfg.model.roi_pooler = CfgNode()
    cfg.model.roi_pooler.featmaps = ['0']
    cfg.model.mask_roi_pooler = CfgNode()
    cfg.model.mask_roi_pooler.featmaps = ['0']

    ####################################################################################################################
    cfg.training = CfgNode()
    cfg.training.checkpoint_every = 100
    cfg.training.visualise_every = 5
    cfg.training.show_visualisations = False

    # glob pattern specifying the json COCO-style files describing the data used in training
    cfg.training.n_epochs = 1_000
    cfg.training.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    cfg.training.batch_size = 10
    cfg.training.shuffle_every_epoch = True

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
    cfg.training.sched.kind = 'OneCycle'  # ('OneCycle', 'Linear', 'Step', 'Exponential', 'None')

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

    cfg.training.early_stopping = CfgNode()
    cfg.training.early_stopping.criteria = 'mAP'
    cfg.training.early_stopping.n_epochs = 5
    cfg.training.early_stopping.less_is_better = False
    cfg.training.early_stopping.thresh = 0.0

    cfg.xval = CfgNode()
    cfg.xval.n_folds = 5

    return cfg


def finalise(cfg: CfgNode):
    if cfg.action == 'contrastive':
        assert cfg.data.is_classified_images, 'contrastive learning needs plain images as input (not JSON files)'
    # data spec
    assert cfg.data.pattern is not None, 'cfg.data.pattern must be specified and must be a string or a list of strings.'
    if isinstance(cfg.data.pattern, str):
        cfg.data.pattern = [cfg.data.pattern]

    if cfg.data.max_number_images is not None:
        assert isinstance(cfg.data.max_number_images, int), 'cfg.data.max_number_images, if set, must be an integer.'

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


def as_hyperparams(cfg: CfgNode) -> dict:
    rv = dict()
    rv['model.architecture'] = 'Mask R-CNN'  # obviously...
    rv['model.backbone'] = cfg.model.backbone.kind
    state_name = cfg.model.state or 'COCO'
    rv['model.pretrained'] = 'No' if cfg.model.state is not None else f'Yes - {state_name}'
    if cfg.model.backbone.kind == 'resnet':
        rv['model.resnet.n'] = cfg.model.backbone.resnet.n
    rv['model.backbone.trainable_layers'] = cfg.model.backbone.trainable_layers
    rv['model.rpn.batch'] = cfg.model.rpn_batch_size_per_image
    # for k, v in dict(cfg.data).items():
    #     rv[f'data/{k}'] = v
    rv['training.n_epochs'] = cfg.training.n_epochs
    rv['training.batch_size'] = cfg.training.batch_size
    rv['training.device'] = cfg.training.device
    return rv
