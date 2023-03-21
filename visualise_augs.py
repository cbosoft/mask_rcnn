from copy import deepcopy

from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
import cv2
import numpy as np

from mask_rcnn.config import get_config
from mask_rcnn.dataset import build_dataset
from mask_rcnn.augmentations.random_flip import RandomFlip
from mask_rcnn.augmentations.colorjitter import ColorJitter
from mask_rcnn.classes import bgr_colour_for_class

cfg = get_config()
cfg.data.pattern = ['D:\\Data\\DF\\TaIs_classified_filtered_DF_subset.json']
ds = build_dataset(cfg)

augmentations = {
    'flip': RandomFlip(is_demo=True),
    'cj-default': ColorJitter(),
    'brightness 0.1': ColorJitter(0.1, 0.0, 0.0, 0.0),
    'brightness 0.5': ColorJitter(0.5, 0.0, 0.0, 0.0),
    'contrast 0.1': ColorJitter(0.0, 0.1, 0.0, 0.0),
    'contrast 0.5': ColorJitter(0.0, 0.5, 0.0, 0.0),
    'colour all 0.5': ColorJitter(0.0, 0.5, 0.5, 0.5),
}


def vis_img_tgt(img, tgt):
    img = (img.permute(1, 2, 0) * 255.).cpu().numpy().astype('uint8').copy()
    plt.imshow(img)

    for mask, lbl in zip(tgt['masks'], tgt['labels']):
        mask = mask.cpu().numpy()
        contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0][0]
        x = contours[:, 0, 0]
        y = contours[:, 0, 1]
        xy = np.array(list(zip(x, y)))
        ax = plt.gca()
        ax: plt.Axes
        colour_bgr = bgr_colour_for_class(lbl)
        colour_rgb = [v/255. for v in colour_bgr[::-1]]
        ax.patches.append(Polygon(xy, alpha=0.5, facecolor=colour_rgb))


N = 5
indices = list(range(len(ds)))
np.random.shuffle(indices)

for aug_n, aug in augmentations.items():
    fig, axes = plt.subplots(N, 2, figsize=(6, 4*N))
    for ax in axes.flatten():
        plt.sca(ax)
        plt.axis('off')

    for i in range(N):
        item = ds[indices[i]]
        img = item['image']
        tgt = item['target']
        plt.sca(axes[i, 0])
        if i == 0:
            plt.title('original')
        vis_img_tgt(img, tgt)

        tfmed_img, tfmed_tgt = aug([img], [tgt])
        tfmed_img, tfmed_tgt = tfmed_img[0], tfmed_tgt[0]
        plt.sca(axes[i, 1])
        if i == 0:
            plt.title(f'transformed: {aug_n}')
        vis_img_tgt(tfmed_img, tfmed_tgt)

    plt.tight_layout()
    plt.show()
