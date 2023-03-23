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


def interpret_coco_data(coco_eval, is_precision=True, iou_threshold=None, area_range='all', max_detections=100, category_id=None):
    if category_id is None:
        category_id = slice(0, -1)
    else:
        category_id = int(category_id)
        assert category_id > 0
        category_id -= 1
    p = coco_eval.params

    aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == area_range]
    mind = [i for i, mDet in enumerate(p.maxDets) if mDet == max_detections]

    # dimension of precision: [TxRxKxAxM], dimension of recall: [TxKxAxM]
    data = coco_eval.eval['precision' if is_precision else 'recall']
    if iou_threshold is not None:
        t = np.where(iou_threshold == p.iouThrs)[0]
        data = data[t]
    data = data[..., category_id, aind, mind]

    mean = np.mean(data[data >= 0])
    if not np.isfinite(mean):
        mean = -1.0

    ap_ar = 'AP' if is_precision else 'AR'
    iou_pfx, iou_sfx = ('', str(int(iou_threshold*100))) if iou_threshold else ('m', '')
    size_sfx = '' if area_range == 'all' else area_range[0].upper()
    cat_sfx = f'_c{category_id+1:02}' if isinstance(category_id, int) else ''
    return f'{iou_pfx}{ap_ar}{iou_sfx}{size_sfx}{cat_sfx}', mean


def coco_eval_datasets(gt, dt):
    coco_gt = COCO()
    coco_gt.dataset = gt
    coco_gt.createIndex()

    coco_dt = COCO()
    coco_dt.dataset = dt
    coco_dt.createIndex()

    coco_eval = COCOeval(coco_gt, coco_dt)
    coco_eval.evaluate()

    coco_eval.accumulate()
    results = {}
    results.update([
        interpret_coco_data(coco_eval, True, None, 'all', 100, None),
        interpret_coco_data(coco_eval, True, 0.5, 'all', 100, None),
        interpret_coco_data(coco_eval, True, None, 'small', 100, None),
        interpret_coco_data(coco_eval, True, None, 'medium', 100, None),
        interpret_coco_data(coco_eval, True, None, 'large', 100, None),
        *[
            interpret_coco_data(coco_eval, True, None, 'all', 100, cat)
            for cat in coco_eval.params.catIds
        ],
        interpret_coco_data(coco_eval, False, None, 'all', 100, None),
        interpret_coco_data(coco_eval, False, None, 'small', 100, None),
        interpret_coco_data(coco_eval, False, None, 'medium', 100, None),
        interpret_coco_data(coco_eval, False, None, 'large', 100, None),
        *[
            interpret_coco_data(coco_eval, False, None, 'all', 100, cat)
            for cat in coco_eval.params.catIds
        ],
    ])

    # Strip categories from all predictions and truth (well, set to cat1)
    dt['categories'] = gt['categories'] = [cat for cat in gt['categories'] if cat['id'] == 1]
    for ann in gt['annotations']:
        ann['category_id'] = 1
    for ann in dt['annotations']:
        ann['category_id'] = 1

    coco_gt = COCO()
    coco_gt.dataset = gt
    coco_gt.createIndex()

    coco_dt = COCO()
    coco_dt.dataset = dt
    coco_dt.createIndex()

    coco_eval = COCOeval(coco_gt, coco_dt)
    coco_eval.evaluate()

    coco_eval.accumulate()
    results.update([
        ('mAP-ca', interpret_coco_data(coco_eval, True, None, 'all', 100, None)[1]),
        ('AP50-ca', interpret_coco_data(coco_eval, True, 0.5, 'all', 100, None)[1]),
        ('mAR-ca', interpret_coco_data(coco_eval, False, None, 'all', 100, None)[1]),
    ])
    return results
