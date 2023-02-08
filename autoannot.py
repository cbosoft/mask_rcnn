import os
import json
from datetime import datetime

from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import torch

from email_notifier import EmailNotifier

from mask_rcnn.config import get_config, finalise
from mask_rcnn.model import build_model
from mask_rcnn.util import visualise_output, hex2rgb, ensure_dir, today, imread
from mask_rcnn.dataset.coco import COCODataset
from mask_rcnn.particle import Particles, ParticleConstructionError
from mask_rcnn.progress_bar import progressbar

import cv2

if __name__ == '__main__':
    # with EmailNotifier(message='Mask R-CNN inference complete'):
    if True:
        """
        To run inference, we need three things:
          1. Config
          2. Model state
          3. Some data
        
        The config file defines the parameters of the model so it can be built, then the state defines the weights and biases of the model.
        """
        MODEL_STATE_PATH = 'training_results/2022-10-24_12-03-25/model_state_at_epoch=500.pth'
        DATASET_PATH = '/Volumes/1TB Data/Data/DF/Kinetics.json'

        CONFIG_FILE = os.path.join(os.path.dirname(MODEL_STATE_PATH), 'config.yaml')
        INF_DATASET_PATH = datetime.now().strftime('inference_DF_%Y-%m-%d_%H-%M-%S.json')

        """
        Create the model, load state.
        """

        cfg = get_config()
        cfg.merge_from_file(CONFIG_FILE)
        cfg.model.state = MODEL_STATE_PATH
        cfg.data.pattern = DATASET_PATH
        cfg.training.device = 'cpu'
        finalise(cfg)
        model = build_model(cfg)
        model.eval()

        """
        Keep the same colours for each label as in CVAT
        """
        # COLOUR_BY_LABEL = {
        #     1: hex2rgb('#33DDFF', 'bgr'),  # elongated
        #     2: hex2rgb('#FA3253', 'bgr'),  # regular
        #     3: hex2rgb('#34D1B7', 'bgr'),  # spherical/circular
        #     4: hex2rgb('#D7B804', 'bgr'),  # agglomerate
        #     5: hex2rgb('#DDFF33', 'bgr'),  # v small
        # }

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

        """
        Apply the model to the specified dataset. Normally, we run models on images it has not seen before (i.e. ones not used in training).
        
        The dataset object reads in the json files we get from CVAT, and normally it ignores images which don't have any annotations (as these are not useful in training). We can, however, run inference on these un-annotated images. By running inference on these images only, we are running inference on images the model has never seen before. 
        """

        with open(DATASET_PATH) as f:
            data = json.load(f)
        
        # data['annotations'] = annots = []
        for image in tqdm(data['images'][:25]):
            image_id = image['id']
            sx = image['width']/256.
            sy = image['height']/256.
            file_name = os.path.join(DATASET_PATH[:-5], image['file_name'])
            image = cv2.imread(file_name, cv2.IMREAD_COLOR)
            assert image is not None, f'Reading image "{file_name}" failed.'
            image = cv2.resize(image, (256, 256))
            image = torch.tensor(image).permute(2, 0, 1)
            image = (image/255.).to(torch.float)

            out = model([image])[0]
            masks = out['masks'].detach().cpu().numpy()
            labels = out['labels'].detach().cpu().numpy()
            scores = out['scores'].detach().cpu().numpy()
            for mask, lbl, score in zip(masks, labels, scores):
                if score < 0.5:
                    break
                mask = (mask[0] > 0.5).astype(np.uint8)*255
                contour = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0][0]
                annot = np.zeros(contour.size)
                x1 = contour[:, 0, 0].min()
                x2 = contour[:, 0, 0].max()
                y1 = contour[:, 0, 1].min()
                y2 = contour[:, 0, 1].max()
                w = x2 - x1
                h = y2 - y1
                area = float(w*h)
                if not int(area):
                    continue
                annot[::2] = contour[:, 0, 0]*sx
                annot[1::2] = contour[:, 0, 1]*sy
                annot = [int(a) for a in annot]
                annots.append(dict(
                    id=len(annots),
                    image_id=image_id,
                    category_id=int(lbl),
                    bbox=[int(x1), int(y1), int(w), int(h)],
                    segmentation=[annot],
                    area=float(w*h),
                    iscrowd=0,
                ))
                
        with open(DATASET_PATH, 'w') as f:
            json.dump(data, f)