import os

import numpy as np
from matplotlib import pyplot as plt

from mask_rcnn.config import get_config, finalise
from mask_rcnn.model import build_model
from mask_rcnn.util import visualise_output, hex2rgb, ensure_dir, today, imread
from mask_rcnn.dataset.coco import COCODataset
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
    CONFIG_FILE = 'training_results/2022-09-02_11-07-02/config.yaml'
    MODEL_STATE_PATH = 'training_results/2022-09-02_11-07-02/model_state_at_epoch=50.pth'
    DATASET_PATH = 'E:/Data/Karen Robertson data/instances_default.json'

    """
    Create the model, load state.
    """

    cfg = get_config()
    cfg.merge_from_file(CONFIG_FILE)
    cfg.model.state = MODEL_STATE_PATH
    cfg.data.pattern = DATASET_PATH
    finalise(cfg)
    model = build_model(cfg)
    model.eval()

    """
    Keep the same colours for each label as in CVAT
    """
    COLOUR_BY_LABEL = {
        1: hex2rgb('#33DDFF', 'bgr'),  # elongated
        2: hex2rgb('#FA3253', 'bgr'),  # regular
        3: hex2rgb('#34D1B7', 'bgr'),  # spherical/circular
        4: hex2rgb('#D7B804', 'bgr'),  # agglomerate
        5: hex2rgb('#DDFF33', 'bgr'),  # v small
    }

    """
    Apply the model to the specified dataset. Normally, we run models on images it has not seen before (i.e. ones not used in training).
    
    The dataset object reads in the json files we get from CVAT, and normally it ignores images which don't have any annotations (as these are not useful in training). We can, however, run inference on these un-annotated images. By running inference on these images only, we are running inference on images the model has never seen before. 
    """

    dataset = COCODataset.from_config(cfg, filter_images='annot')

    dsname, _ = os.path.splitext(os.path.basename(DATASET_PATH))
    output_dir = ensure_dir(f'Inference Output/{today()}_{dsname}')

    # n_to_vis = 1
    # for i, batch in enumerate(dataset):
    #     if i >= n_to_vis:
    #         break
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
    #     out = model(inp)[0]
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

    particles = Particles()
    for batch in progressbar(dataset):
        inp = batch['image'].unsqueeze(0)
        oimg = imread(batch['source'])
        rshape = inp.shape[-2:]
        ishape = oimg.shape[-2:]
        sy = ishape[0]/rshape[0]
        sx = ishape[1]/rshape[1]

        out = model(inp)[0]
        masks = out['masks']
        scores = out['scores']
        masks = ((masks.squeeze().detach().cpu().numpy() > 0.5) * 255.).astype(np.uint8)
        scores = scores.detach().cpu().numpy()
        for mask, score in zip(masks, scores):
            if score < 0.75:
                break
            contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
            if not len(contours):
                print('empty annot?')
                continue
            contour = contours[0]
            contour = contour.astype(float)
            contour[..., 0] *= sx
            contour[..., 1] *= sy
            contour = contour.astype(np.int32)
            try:
                particles.add(batch['source'], cv2.imread(batch['source'], cv2.IMREAD_GRAYSCALE), contour, 1.26, score, 5)
            except ParticleConstructionError as e:
                print(e)

    lengths = [p.length for p in particles]
    dist, edges = np.histogram(lengths, density=True)
    bins = (edges[1:]+edges[:-1])*0.5
    plt.plot(bins, dist)
    plt.show()
