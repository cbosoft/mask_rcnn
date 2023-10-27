from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
import cv2
import numpy as np

import mlflow

from .classes import bgr_colour_for_class


def visualise_valid_batch(images, targets, outputs, should_show_visualisations: bool, output_dir: str, epoch: int, prefix: str):

    for i, (image, target, output) in enumerate(zip(images, targets, outputs)):
        _, axes = plt.subplots(nrows=1, ncols=3, figsize=(3*4, 4))
        for ax in axes:
            plt.sca(ax)
            plt.axis('off')

        # prep image for vis
        image = (image.permute(1, 2, 0) * 255.).cpu().numpy().astype('uint8')

        plt.sca(axes[0])
        plt.title('Raw')
        plt.imshow(image)

        # draw prediction
        plt.sca(axes[2])
        plt.title('Prediction')
        plt.imshow(image)
        omasks, oscores, olabels = output['masks'], output['scores'], output['labels']
        msl = list(zip(*filter(lambda mbs: mbs[1] > 0.1, zip(omasks, oscores, olabels))))
        if msl:
            omasks, oscores, olabels = msl
        else:
            omasks, oscores, olabels = [], [], []

        for score, mask, lbl in zip(oscores, omasks, olabels):
            mask = (mask[0].cpu().numpy() > 0.5).astype(np.uint8)
            if np.all(mask == 0): continue
            contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0][0]
            x = contours[:, 0, 0]
            y = contours[:, 0, 1]
            xy = np.array(list(zip(x, y)))
            colour = bgr_colour_for_class(lbl)
            colour = [v/255. for v in colour[::-1]]
            axes[2].add_patch(Polygon(xy, alpha=0.5, facecolor=colour))

        # draw ground truth
        plt.sca(axes[1])
        plt.title('Ground Truth')
        plt.imshow(image)
        for mask, lbl in zip(target['masks'], target['labels']):
            mask = mask.cpu().numpy()
            contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0][0]
            x = contours[:, 0, 0]
            y = contours[:, 0, 1]
            xy = np.array(list(zip(x, y)))
            colour = bgr_colour_for_class(lbl)
            colour = [v/255. for v in colour[::-1]]
            axes[1].add_patch(Polygon(xy, alpha=0.5, facecolor=colour))

        plt.tight_layout()
        if should_show_visualisations:
            plt.show()
        fn = f'{output_dir}/{prefix}seg_{i}_epoch={epoch}.jpg'
        plt.savefig(fn)
        mlflow.log_artifact(fn)
        plt.close()
