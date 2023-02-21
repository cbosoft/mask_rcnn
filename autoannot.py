import os
import json

from tqdm import tqdm
import numpy as np
import torch

# from email_notifier import EmailNotifier

from mask_rcnn.config import get_config, finalise
from mask_rcnn.model import build_model

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
        MODEL_STATE_PATH = 'training_results/2022-10-13_11-01-20/model_state_at_epoch=500.pth'
        DATASET_PATH = r'D:\Data\Industrial\G.json'

        CONFIG_FILE = os.path.join(os.path.dirname(MODEL_STATE_PATH), 'config.yaml')

        """
        Create the model, load state.
        """

        cfg = get_config()
        cfg.merge_from_file(CONFIG_FILE)
        cfg.model.state = MODEL_STATE_PATH
        cfg.data.pattern = DATASET_PATH
        # cfg.training.device = 'cpu'
        finalise(cfg)
        model = build_model(cfg)
        model.eval()

        """
        Apply the model to the specified dataset. Normally, we run models on images it has not seen before (i.e. ones not used in training).
        
        The dataset object reads in the json files we get from CVAT, and normally it ignores images which don't have any annotations (as these are not useful in training). We can, however, run inference on these un-annotated images. By running inference on these images only, we are running inference on images the model has never seen before. 
        """

        with open(DATASET_PATH) as f:
            data = json.load(f)
        
        annots_by_id = {}
        for annot in data['annotations']:
            im_id = annot['image_id']
            if im_id not in annots_by_id:
                annots_by_id[im_id] = []
            annots_by_id[im_id].append(annot)
        
        for image in tqdm(data['images']):
            image_id = image['id']
            annots_by_id[image_id] = []  # clear annotations from this image
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
                annots_by_id[image_id].append(dict(
                    id=-1,
                    image_id=image_id,
                    category_id=int(lbl),
                    bbox=[int(x1), int(y1), int(w), int(h)],
                    segmentation=[annot],
                    area=float(w*h),
                    iscrowd=0,
                ))
        
        data['annotations'] = []
        for annots in annots_by_id.values():
            for annot in annots:
                annot['id'] = len(data['annotations'])
                data['annotations'].append(annot)
                
        with open(DATASET_PATH, 'w') as f:
            json.dump(data, f)
