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
    _ = cfg  # TODO: build up from config
    anchor_generator = AnchorGenerator(
        sizes=(
            (32, 64, 128, 256, 512),
            (32, 64, 128, 256, 512),
            (32, 64, 128, 256, 512),
            (32, 64, 128, 256, 512),
            (32, 64, 128, 256, 512),
        ),
        aspect_ratios=(
            (0.5, 1.0, 2.0),
            (0.5, 1.0, 2.0),
            (0.5, 1.0, 2.0),
            (0.5, 1.0, 2.0),
            (0.5, 1.0, 2.0),
        ))
    return anchor_generator


def build_box_roi_pooler(cfg: CfgNode) -> MultiScaleRoIAlign:
    _ = cfg  # TODO: build up from config
    return MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)


def build_mask_roi_pooler(cfg: CfgNode) -> MultiScaleRoIAlign:
    _ = cfg  # TODO: build up from config
    return MultiScaleRoIAlign(featmap_names=['0'], output_size=14, sampling_ratio=2)


def build_model(cfg: CfgNode) -> MaskRCNN:
    model = MaskRCNN(
        backbone=build_backbone(cfg),
        num_classes=cfg.model.n_classes,
        rpn_anchor_generator=build_rpn_anchor_generator(cfg),
        box_roi_pool=build_box_roi_pooler(cfg),
        mask_roi_pool=build_mask_roi_pooler(cfg)
    )

    # TODO initialise weights or load state

    return model
