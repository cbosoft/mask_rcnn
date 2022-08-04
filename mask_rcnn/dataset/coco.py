from glob import glob
import json
from typing import List
import os

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset as _TorchDataset

from ..config import CfgNode


class COCO_Annotation:

    def __init__(self, *, id, image_id, category_id, segmentation, area, bbox, iscrowd, attributes):
        self.id = id
        self.image_id = image_id
        self.category_id = category_id
        self.segmentation = []
        for seg in segmentation:
            seg = np.array(seg)
            sx = seg[::2]
            sy = seg[1::2]
            self.segmentation.append(np.array(list(zip(sx, sy)), dtype=np.int32))
        self.area = area
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h
        self.bbox = (x1, y1, x2, y2)
        self.iscrowd = iscrowd
        self.attributes = attributes

    def get_mask(self, size, scale):
        w, h = size
        sx, sy = scale
        mask = np.zeros((w, h), dtype='uint8')
        seg = []
        for segi in self.segmentation:
            segi = np.copy(segi).astype(float)
            segi[:, 0] *= sx
            segi[:, 1] *= sy
            seg.append(segi.astype(np.int32))
        cv2.drawContours(mask, seg, -1, 1, -1)
        return mask


class COCO_Image:

    size = width, height = 256, 256

    def __init__(self, *, file_name, width, height, **_):
        self.file_name: str = file_name
        self.orig_width = width
        self.orig_height = height
        self.scale = self.scale_x, self.scale_y = self.width/self.orig_width, self.height/self.orig_height
        self.annotations: List[COCO_Annotation] = []

        image = cv2.imread(self.file_name, cv2.IMREAD_COLOR)
        image = cv2.resize(image, self.size)
        self.image = torch.tensor(image).permute(2, 0, 1)
        self.target_dict = None

    def scale_bbox(self, bbox):
        x1, y1, x2, y2 = bbox
        return self.scale_x*x1, self.scale_y*y1, self.scale_x*x2, self.scale_y*y2

    def get_target_dict(self) -> dict:
        if self.target_dict is None:
            boxes = torch.tensor([self.scale_bbox(a.bbox) for a in self.annotations]).float()
            labels = torch.tensor([a.category_id for a in self.annotations], dtype=torch.int64)
            masks = torch.tensor(np.array([a.get_mask(self.size, self.scale) for a in self.annotations]), dtype=torch.uint8)
            self.target_dict = dict(boxes=boxes, labels=labels, masks=masks)
        return self.target_dict


class COCODataset(_TorchDataset):

    def __init__(self, images: List[COCO_Image], transforms=None):
        self.images = images
        self.transforms = transforms

    @classmethod
    def from_config(cls, cfg: CfgNode):
        images_by_id = {}
        fns = []
        for pattern in cfg.data.pattern:
            fns.extend(glob(pattern))

        for fn in fns:
            bn = os.path.dirname(fn)
            with open(fn) as f:
                coco_dataset = json.load(f)

            for im_data in coco_dataset['images']:
                im_id = im_data['id']
                im_data['file_name'] = os.path.join(bn, im_data['file_name'])
                images_by_id[im_id] = COCO_Image(**im_data)

            for ann_data in coco_dataset['annotations']:
                im_id = ann_data['image_id']
                images_by_id[im_id].annotations.append(COCO_Annotation(**ann_data))

        # n_categories = max([max([a.category_id for a in i.annotations]) for i in images.values()])+1

        return cls([im for im in images_by_id.values() if im.annotations])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img_data = self.images[i]
        tgt = img_data.get_target_dict()
        img = img_data.image
        if self.transforms:
            img = self.transforms(img)
        img = (img/255.).to(torch.float)
        # TODO resize tgt masks/boxes too?
        return dict(image=img, target=tgt, source=img_data.file_name)
