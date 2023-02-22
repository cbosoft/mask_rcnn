from typing import List, Dict

import cv2
import numpy as np
import torch
import pycocotools
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval, Params


def noop(*args, **kwargs):
    pass


# Monkeypatch overwrite of print function:
# remove excessive printing in pyocotools.
pycocotools.coco.print = noop
pycocotools.cocoeval.print = noop


def update_coco_datasets_from_batch(
        coco_images: dict,
        coco_gt_anns: List[dict],
        coco_dt_anns: List[dict],
        tgt: List[Dict[str, torch.Tensor]],
        out: List[Dict[str, torch.Tensor]]):

    assert len(tgt) == len(out)
    for tgt_data, out_data in zip(tgt, out):
        image_id = tgt_data['id']
        coco_images[image_id] = dict(
            id=tgt_data['id'],
            file_name=tgt_data['file_name'],
            width=tgt_data['width'],
            height=tgt_data['height'],
        )

        for mask, label in zip(tgt_data['masks'], tgt_data['labels']):
            mask = mask.cpu().numpy()
            contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

            assert contours, (contours, mask.min(), mask.max(), mask.shape)
            contours = contours[0]

            sx = contours[:, :, 0]
            sy = contours[:, :, 1]
            sx, sy = sx.flatten(), sy.flatten()
            x1, x2 = np.min(sx), np.max(sx)
            y1, y2 = np.min(sy), np.max(sy)
            w = x2 - x1
            h = y2 - y1
            area = w*h
            segmentation = np.zeros(len(sx)*2)
            segmentation[::2] = sx
            segmentation[1::2] = sy
            assert len(segmentation) > 4
            annotation = dict(
                id=-1,
                image_id=image_id,
                category_id=int(label),
                area=int(area),
                segmentation=[[int(f) for f in segmentation]],
                bbox=[x1, y1, x2, y2],
                iscrowd=0
            )
            coco_gt_anns.append(annotation)

        for mask, label, score in zip(out_data['masks'], out_data['labels'], out_data['scores']):
            mask = (mask[0].cpu().numpy() > 0.5).astype(np.uint8)
            contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

            if not contours:
                continue

            assert contours, (contours, mask.min(), mask.max(), mask.shape)
            contours = contours[0]

            sx = contours[:, :, 0]
            sy = contours[:, :, 1]
            sx, sy = sx.flatten(), sy.flatten()
            x1, x2 = np.min(sx), np.max(sx)
            y1, y2 = np.min(sy), np.max(sy)
            w = x2 - x1
            h = y2 - y1
            area = w*h
            segmentation = np.zeros(len(sx)*2)
            segmentation[::2] = sx
            segmentation[1::2] = sy

            if len(segmentation) < 6:
                continue

            annotation = dict(
                id=-1,
                image_id=image_id,
                category_id=int(label),
                area=int(area),
                segmentation=[[int(f) for f in segmentation]],
                bbox=[x1, y1, x2, y2],
                iscrowd=0,
                score=float(score)
            )
            coco_dt_anns.append(annotation)



def coco_eval_datasets(gt, dt):
    coco_gt = COCO()
    coco_gt.dataset = gt
    coco_gt.createIndex()

    coco_dt = COCO()
    coco_dt.dataset = dt
    coco_dt.createIndex()

    coco_eval = COCOeval(coco_gt, coco_dt)
    coco_eval.evaluate()

    p = Params()
    p.areaRng = [[0, 1e10]]
    p.areaRngLbl = ['all']
    p.maxDets = [100]
    p.imgIds = sorted(coco_gt.getImgIds())
    p.catIds = list(range(1, 11))

    coco_eval.accumulate(p)

    data = dict()
    APs = []
    for i, iou_thresh in enumerate(p.iouThrs):
        k = f'AP{int(iou_thresh*100.)}'
        v = coco_eval.eval['precision'][i, :, :, 0, 0]
        v = v[v >= 0.0]
        v = np.mean(v)
        if not np.isfinite(v):
            v = 0.0
        data[k] = v
        APs.append(v)
    data['mAP'] = np.mean(APs)
    return data
