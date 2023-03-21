from glob import glob
import json
from typing import List
import os
from copy import deepcopy

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset as _TorchDataset

from ..config import CfgNode
from ..progress_bar import progressbar


class COCO_Annotation:

    def __init__(self, *, id, image_id, category_id, segmentation, area, bbox, iscrowd, attributes=None):
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

    def get_mask(self, size, scale=(1.0, 1.0)):
        sx, sy = scale
        mask = np.zeros(size, dtype='uint8')
        seg = []
        for segi in self.segmentation:
            segi = np.copy(segi).astype(float)
            segi[:, 0] *= sx
            segi[:, 1] *= sy
            seg.append(segi.astype(np.int32))
        cv2.drawContours(mask, seg, -1, 1, -1)
        return mask


class COCO_Image:

    def __init__(self, *, id, file_name, width, height, **_):
        self.id = id
        self.file_name: str = file_name
        self.orig_width = width
        self.orig_height = height
        self.annotations: List[COCO_Annotation] = []
        self.target_dict = None

    def _get_image(self) -> torch.Tensor:
        image = cv2.imread(self.file_name, cv2.IMREAD_COLOR)
        assert image is not None, f'Reading image "{self.file_name}" failed.'
        return torch.tensor(image).permute(2, 0, 1)

    @property
    def image(self):
        return self._get_image()

    def get_target_dict(self) -> dict:
        if self.target_dict is None:
            boxes = torch.tensor([a.bbox for a in self.annotations]).float()
            labels = torch.tensor([a.category_id for a in self.annotations], dtype=torch.int64)
            masks = torch.tensor(np.array([a.get_mask((self.orig_height, self.orig_width)) for a in self.annotations]), dtype=torch.uint8)
            self.target_dict = dict(boxes=boxes, labels=labels, masks=masks, id=id, file_name=self.file_name, width=self.orig_width, height=self.orig_height)
        return self.target_dict


class COCODataset(_TorchDataset):

    def __init__(self, images: List[COCO_Image], max_n_images: int):
        self.images = images if max_n_images is not None else images[:max_n_images]

    @staticmethod
    def get_dataset_files(cfg):
        fns = []
        for pattern in cfg.data.pattern:
            fns.extend(glob(pattern))
        return fns

    @classmethod
    def from_config(cls, cfg: CfgNode, filter_images='empty'):
        """
        Create dataset from COCO-format json file(s)

        :param cfg:
        :param filter_images: Whether to filter the files. 'none' for no filtering, 'empty' to remove empty (default), 'annot' to remove annotated images. The last two are useful for training and validation respectively.
        :return:
        """
        images = []
        fns = cls.get_dataset_files(cfg)
        
        assert fns, f'No datasets found! Double check cfg.data.pattern: {cfg.data.pattern}'

        for fn in fns:
            dn = os.path.dirname(fn)
            with open(fn) as f:
                coco_dataset = json.load(f)

            images_by_orig_id = {}

            for im_data in progressbar(coco_dataset['images'], unit='images', desc='1/2'):
                # when combining dataset json files, IDs are not unique and need to be recalculated.
                orig_im_id = im_data['id']
                im_data['id'] = im_id = len(images) + len(images_by_orig_id)
                im_data['file_name'] = os.path.join(dn, im_data['file_name'])
                im_data['file_name'] = im_data['file_name'].replace('\\', '/')
                images_by_orig_id[orig_im_id] = COCO_Image(**im_data)

            for ann_data in progressbar(coco_dataset['annotations'], unit='annotations', desc='2/2'):
                # update annotation to refer to recalculated image IDs
                orig_im_id = ann_data['image_id']
                ann_data['image_id'] = images_by_orig_id[orig_im_id].id
                images_by_orig_id[orig_im_id].annotations.append(COCO_Annotation(**ann_data))
            
            orig_n_images = len(images_by_orig_id)
            if filter_images == 'none':
                images_filtered = list(images_by_orig_id.values())
            elif filter_images == 'empty':
                images_filtered = [im for im in images_by_orig_id.values() if im.annotations]
            elif filter_images == 'annot':
                images_filtered = [im for im in images_by_orig_id.values() if not im.annotations]
            else:
                raise ValueError(f'Didn\'t understand value for $filter_images ({filter_images}), should be one of "none", "empty", or "annot"')
            n_images = len(images_filtered)
            if n_images < orig_n_images:
                print(f'Filtered {orig_n_images - n_images} {filter_images} images from dataset "{fn}" ({n_images} remain).')

            images.extend(images_filtered)

        print(f'{len(images)} total images.')

        return cls(images, cfg.data.max_number_images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img_data = self.images[i]
        tgt = img_data.get_target_dict()
        img = img_data.image
        img = (img/255.).to(torch.float)
        # Images are read in fresh every time, but targets are kept in-memory
        # Return copy of targets so transforms can be done in-place.
        return dict(image=img, target=deepcopy(tgt), source=img_data.file_name)
