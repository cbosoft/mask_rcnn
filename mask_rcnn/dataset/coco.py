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

    def get_mask(self, width, height):
        mask = np.zeros((height, width), dtype='uint8')
        cv2.drawContours(mask, self.segmentation, -1, self.category_id, -1)
        return mask


class COCO_Image:

    def __init__(self, *, file_name, width, height, **_):
        self.file_name: str = file_name
        self.width = width
        self.height = height
        self.annotations: List[COCO_Annotation] = []

    def get_target_dict(self) -> dict:
        boxes = torch.tensor([a.bbox for a in self.annotations]).float()
        labels = torch.tensor([a.category_id for a in self.annotations], dtype=torch.int64)
        masks = torch.tensor([a.get_mask(self.width, self.height) for a in self.annotations], dtype=torch.uint8)
        return dict(boxes=boxes, labels=labels, masks=masks)

    def get_image(self):
        im = cv2.imread(self.file_name, cv2.IMREAD_GRAYSCALE)
        return torch.tensor(im)


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
        img = self.images[i]
        tgt = img.get_target_dict()
        img = img.get_image()
        if self.transforms:
            img = self.transforms(img)
        img = (img/255.).to(torch.float)
        img = torch.stack([img, img, img])
        # TODO resize tgt masks/boxes too?
        return dict(image=img, target=tgt)
