from typing import Tuple, Callable, List
import os
from glob import glob

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset as TorchDataset


from ..config import CfgNode


class ClassifiedImage:

    def __init__(self, filename: str, cls: str):
        self.filename = filename
        self.cls = cls


class ImagesDataset(TorchDataset):

    EXTENSIONS = {'.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff'}

    def __init__(self, images: List[ClassifiedImage], device):
        self.images = images
        self.device = device
        all_classes = sorted(set(im.cls for im in images))
        self.idx_by_class = {c: i for i, c in enumerate(all_classes)}
        print(f'{len(images)} images, {len(all_classes)} classes, on {device}')
        print(self.idx_by_class)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i) -> dict:
        im_fn_cls = self.images[i]
        cls_index = self.idx_by_class[im_fn_cls.cls]
        image = cv2.imread(im_fn_cls.filename, cv2.IMREAD_COLOR)
        assert image is not None, f'Reading image "{self.file_name}" failed.'
        image = image.astype(float)/255.0
        image = torch.tensor(image).float().permute(2, 0, 1).to(self.device)
        return dict(image=image, cls=cls_index)

    @classmethod
    def from_config(cls, cfg: CfgNode):
        image_files = cls.get_dataset_files(cfg.data.pattern)
        classifier: Callable[[str], str] = eval(cfg.data.classified_images.classifier, dict(os=os))
        images = []
        for fn in image_files:
            c = classifier(fn)
            image = ClassifiedImage(fn, c)
            images.append(image)
        return cls(images, cfg.training.device)

    @classmethod
    def get_dataset_files(cls, patterns: List[str]):
        fns = []
        for pattern in patterns:
            this_pattern_fns = glob(pattern)
            if not this_pattern_fns:
                print(f'No images found for pattern {pattern}')
            else:
                fns.extend(this_pattern_fns)
        image_fns = [fn for fn in fns if os.path.splitext(fn.lower())[1] in cls.EXTENSIONS]
        assert image_fns
        return image_fns

