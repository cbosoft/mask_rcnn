import torch
from mask_rcnn.metrics.iou import IntersectionOverUnion


def check_iou(a, b, expected_iou):
    iou = IntersectionOverUnion.box_iou(torch.tensor(a), torch.tensor(b))
    assert abs(iou - expected_iou) < 0.01, iou


def test_box_iou_1():
    # Perfect overlap, 100%
    check_iou(a=(0, 0, 1, 1), b=(0, 0, 1, 1), expected_iou=1.0)


def test_box_iou_2():
    # Boxes touching, but 0% overlap
    check_iou(a=(1, 1, 2, 2), b=(0, 0, 1, 1), expected_iou=0.0)


def test_box_iou_3():
    # Boxes overlap 50%
    check_iou(a=(0, 0, 1, 1), b=(0.5, 0, 1.5, 1), expected_iou=1./3.)


def test_box_iou_4():
    # No overlap, boxes 1 unit sq apart
    check_iou(a=(0, 0, 1, 1), b=(2, 2, 3, 3), expected_iou=0.)


def test_box_iou_5():
    # No overlap, boxes 1 unit sq apart
    check_iou(a=(2, 2, 3, 3), b=(0, 0, 1, 1), expected_iou=0.)
