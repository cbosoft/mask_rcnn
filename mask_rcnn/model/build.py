import mldb
import torch
from torch import nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights
)
from torchvision.models.detection import MaskRCNN, backbone_utils
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

from ..config import CfgNode


def build_backbone(cfg: CfgNode) -> nn.Module:
    if cfg.model.backbone.kind == 'resnet':
        n = cfg.model.backbone.resnet.n
        weights = {
            18: ResNet18_Weights, 34: ResNet34_Weights, 50: ResNet50_Weights,
            101: ResNet101_Weights, 152: ResNet152_Weights
        }[n]
        returned_layers = None
        if cfg.model.backbone.returned_layers is not None:
            returned_layers = eval(cfg.model.backbone.returned_layers)
        backbone = backbone_utils.resnet_fpn_backbone(
            backbone_name=f'resnet{n}',
            weights=weights.DEFAULT if cfg.model.backbone.pretrained else None,
            trainable_layers=cfg.model.backbone.trainable_layers if cfg.model.backbone.pretrained else 5,
            returned_layers=returned_layers
        )
    else:
        raise NotImplementedError(f'backbone "{cfg.model.backbone.kind}" not implemented.')
    return backbone


def build_rpn_anchor_generator(cfg: CfgNode) -> nn.Module:
    n_feature_maps = 5  # TODO: depends on model backend
    anchor_generator = AnchorGenerator(
        sizes=tuple([
            tuple(cfg.model.rpn_anchor_generator.sizes)
            for _ in range(n_feature_maps)
        ]),
        aspect_ratios=tuple([
            tuple(cfg.model.rpn_anchor_generator.aspect_ratios)
            for _ in range(n_feature_maps)
        ])
    )
    return anchor_generator


def build_box_roi_pooler(cfg: CfgNode) -> MultiScaleRoIAlign:
    return MultiScaleRoIAlign(
        featmap_names=cfg.model.roi_pooler.featmaps,
        output_size=7,
        sampling_ratio=2
    )


def build_mask_roi_pooler(cfg: CfgNode) -> MultiScaleRoIAlign:
    return MultiScaleRoIAlign(
        featmap_names=cfg.model.mask_roi_pooler.featmaps,
        output_size=14,
        sampling_ratio=2
    )


def build_model(cfg: CfgNode) -> MaskRCNN:
    model = MaskRCNN(
        backbone=build_backbone(cfg),
        num_classes=cfg.model.n_classes,
        rpn_anchor_generator=build_rpn_anchor_generator(cfg),
        box_roi_pool=build_box_roi_pooler(cfg),
        mask_roi_pool=build_mask_roi_pooler(cfg)
    )

    print(model)

    if cfg.model.state:
        if isinstance(cfg.model.state, str):
            state_file = cfg.model.state
        else:
            assert isinstance(cfg.model.state, tuple)
            assert len(cfg.model.state) == 2
            expid, epoch = cfg.model.state
            with mldb.Database() as db:
                state_file = db.get_state_file(expid, epoch)
        model.load_state_dict(torch.load(state_file))

    return model
