from torch import nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

from ..config import CfgNode


def build_backbone(cfg: CfgNode) -> nn.Module:
    _ = cfg  # TODO: build up from config
    backbone = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).features
    backbone.out_channels = 1280
    return backbone


def build_rpn_anchor_generator(cfg: CfgNode) -> nn.Module:
    _ = cfg  # TODO: build up from config
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),))
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
