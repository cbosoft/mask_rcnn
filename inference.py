import os
import json
from glob import glob
from datetime import datetime

import torch
import numpy as np
from matplotlib import pyplot as plt

from mask_rcnn.config import get_config, finalise
from mask_rcnn.model import build_model
from mask_rcnn.util import visualise_output, hex2rgb, ensure_dir, today, imread
# from mask_rcnn.dataset.coco import COCODataset
from mask_rcnn.particle import Particles, ParticleConstructionError
from mask_rcnn.progress_bar import progressbar

import cv2

if __name__ == '__main__':
    """
    To run inference, we need three things:
      1. Config
      2. Model state
      3. Some data
    
    The config file defines the parameters of the model so it can be built, then the state defines the weights and biases of the model.
    """
    MODEL_STATE_PATH = '/home/chris/.MDPC/Model Zoo/Mask_RCNN/crystalline_v1/state.pth'
    # IMAGES_PATH = '/path/to/images/folder'
    IMAGES_PATH = '/home/chris/Documents/Data/DF/Kinetics/DF_DoE_ASP_1Pentanol_1A/images'

    CONFIG_FILE = os.path.join(os.path.dirname(MODEL_STATE_PATH), 'config.yaml')
    INF_DATASET_PATH = datetime.now().strftime('inference_%Y-%m-%d_%H-%M-%S.json')

    """
    Create the model, load state.
    """

    cfg = get_config()
    cfg.merge_from_file(CONFIG_FILE)
    cfg.model.state = MODEL_STATE_PATH
    finalise(cfg)
    model = build_model(cfg)
    model.eval()

    # Matplotlib tab10 palette
    COLOUR_BY_LABEL = {
        1: hex2rgb('#1f77b4', 'bgr'),  # elongated
        2: hex2rgb('#ff7f0e', 'bgr'),  # regular
        3: hex2rgb('#2ca02c', 'bgr'),  # spherical
        4: hex2rgb('#d62728', 'bgr'),  # agglomerate
        5: hex2rgb('#9467bd', 'bgr'),  # v. small

        6: hex2rgb('#8c564b', 'bgr'),  # user 1 ("other" in KaRo)
        7: hex2rgb('#e377c2', 'bgr'),  # user 2 ("plates or platelets" in KaRo)
        8: hex2rgb('#7f7f7f', 'bgr'),  # user 3 ("parallelipipeds" in KaRo)
        9: hex2rgb('#bcbd22', 'bgr'),  # user 4
        10: hex2rgb('#17becf', 'bgr'),  # user 5
    }

    LBL2STR = {
        1: 'Elongated',
        2: 'Regular',
        3: 'Spherical',
        4: 'Agglomerate',
        5: 'V. Small',
        6: 'Other',
        7: 'Plate(let)',
        8: 'Parallelepiped',
        9: 'U4',
        10: 'U5',
    }

    output_dir = ensure_dir(os.path.splitext(INF_DATASET_PATH)[0])

    # # n_to_vis = 100
    # for i, batch in enumerate(dataset):
    #     # if i >= n_to_vis:
    #     #     break
    #     inp = batch['image'].unsqueeze(0)
    #     source = batch['source']
    #     outname = (source
    #                .replace(':', '-')
    #                .replace('/', '-')
    #                .replace('\\', '-')
    #                .replace(' ', '_'))
    #     if len(outname) > 40:
    #         outname = outname[:7] + '...' + outname[-30:]
    #     print(outname)
    #     out = model(inp/255)[0]
    #     masks = out['masks']
    #     scores = out['scores']
    #     labels = out['labels']
    #     if len(masks):
    #         vimg = visualise_output(
    #             inp, masks, scores, labels,
    #             colours=lambda i: COLOUR_BY_LABEL[i],
    #             score_thresh=0.75,
    #         )
    #         cv2.imwrite(f'{output_dir}/{outname}', vimg)

    images = glob(f'{IMAGES_PATH}/*')
    bar = progressbar(images)
    for image_path in bar:
        oimg = cv2.imread(image_path)
        inp = (torch.tensor(oimg).permute(2, 0, 1).float() / 255.).unsqueeze(0)
        oimg_noannot = oimg.copy()
        _, __, *rshape = inp.shape
        *ishape, _ = oimg.shape

        sy = ishape[0]/rshape[0]
        sx = ishape[1]/rshape[1]
        path_parts = image_path.split(os.sep)
        outname = os.sep.join([output_dir, *path_parts[-3:]])
        ensure_dir(os.path.dirname(outname))

        out = model(inp)[0]
        masks = out['masks']

        masks = masks.detach().cpu().numpy()[:, 0]

        scores = out['scores']
        labels = out['labels'].detach().cpu().numpy()

        boxes = out['boxes'].detach().cpu().numpy()

        masks = (masks * 255.).astype(np.uint8)
        scores = scores.detach().cpu().numpy()
        assert len(masks) == len(scores) == len(labels), f'{len(masks)} ==? {len(scores)} ==? {len(labels)}'
        annot_count = 0
        for i, (mask, score, lbl, box) in enumerate(zip(masks, scores, labels, boxes)):
            contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
            assert len(contours)
            contour = contours[0]
            contour = contour.astype(float)
            contour[:, 0, 0] *= sx
            contour[:, 0, 1] *= sy
            contour = contour.astype(np.int32)

            colour = COLOUR_BY_LABEL[int(lbl)]
            cv2.drawContours(oimg, [contour], 0, colour, 3)
            text_x = np.min(contour[:, 0, 0])
            text_y = np.min(contour[:, 0, 1])
            lbl_str = LBL2STR[int(lbl)]
            cv2.putText(oimg, f'{lbl_str} ({score:.3f})', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4, cv2.LINE_AA)
            cv2.putText(oimg, f'{lbl_str} ({score:.3f})', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2, cv2.LINE_AA)
            annot_count += 1
        bar.set_description(f'drew {annot_count} masks on "{outname[-50:]}"')
        assert cv2.imwrite(outname, oimg), f'Writing image to {outname} failed!'

