import os
from typing import List, Dict
import json

import numpy as np

from ..util import imread
from .particle import Particle


class Particles:

    def __init__(self, *particles):
        self.particles: List[Particle] = [*particles]

    def __iter__(self):
        return iter(self.particles)

    def __len__(self):
        return len(self.particles)

    def add(self, fn: str, orig_image: np.ndarray, contour, px2um, score, lbl, on_border_thresh):
        self.particles.append(Particle(fn, orig_image, contour, px2um, score, lbl, on_border_thresh))

    def write_out(self, fn: str, comment=None):
        csv_lines = [','.join(Particle.CSV_HEADER)]
        if comment:
            csv_lines.insert(0, '# ' + comment)
        for particle in sorted(self.particles):
            csv_lines.append(particle.to_csv_line())

        with open(fn, 'w') as f:
            for line in csv_lines:
                f.write(f'{line}\n')

    def split_by_dir(self) -> Dict[str, "Particles"]:
        by_dir = dict()
        for p in self.particles:
            fn = p.image_file_name
            dn = os.path.dirname(fn)
            if dn not in by_dir:
                by_dir[dn] = list()
            by_dir[dn].append(p)

        return {k: Particles(*sorted(v)) for k, v in by_dir.items()}

    def split_by_fn_chunks(self, fnss: List[str]):
        chunks = [list() for _ in fnss]
        for i, fns in enumerate(fnss):
            for p in self.particles:
                if p.image_file_name in fns:
                    chunks[i].append(p)
        return chunks

    def to_dict(self) -> Dict[str, list]:
        values = {k: list() for k in Particle.CSV_HEADER}
        for p in self.particles:
            d = p.to_dict()
            for k in Particle.CSV_HEADER:
                values[k].append(d[k])
        return values

    @staticmethod
    def annot_to_contour(a) -> np.ndarray:
        # a is a list of numbers.
        # every other number is an x-coord, every other is a y-coord.
        x = a[::2]
        y = a[1::2]
        # rearrange so it is in the form of a list of (x, y)
        rv = np.stack([x, y], 1)
        # opencv is wierd and wants a single-item list in the middle so it is a list of single item list of (x, y)
        rv = rv[:, np.newaxis, :]
        # cv wants its contours not in python ints, but specifically int32.
        rv = rv.astype('int32')
        return rv

    @classmethod
    def from_annotations(cls, path_to_json: str, on_border_thresh=5, px2um=1./1.25316) -> "Particles":
        with open(path_to_json) as f:
            data = json.load(f)

        im_by_id = {imd['id']: imd for imd in data['images']}
        ann_by_image = {}
        for ann in data['annotations']:
            im_id = ann['image_id']
            if im_id not in ann_by_image:
                ann_by_image[im_id] = []
            ann_by_image[im_id].append(ann)

        images_with_annotations = set(im_by_id).intersection(ann_by_image)

        particles = Particles()
        for im_id in images_with_annotations:
            imd = im_by_id[im_id]
            im_fn = os.path.join(
                os.path.dirname(path_to_json),
                imd['file_name'])
            oimg = imread(im_fn).cpu().detach().numpy()
            assert im_id in ann_by_image
            for ann in ann_by_image[im_id]:
                ann = ann['segmentation'][0]  # There should only be one segmentation polygon per particle
                contour = cls.annot_to_contour(ann)
                particles.add(imd['file_name'], oimg, contour, px2um, 1.0, ann['category_id'], on_border_thresh)

        return particles
