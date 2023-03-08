from matplotlib import pyplot as plt
import cv2
import numpy as np

from .classes import bgr_colour_for_class


def visualise_valid_batch(images, targets, outputs, should_show_visualisations: bool, output_dir: str, epoch: int, prefix: str):

    if should_show_visualisations:
        fig, axes = plt.subplots(ncols=len(images), squeeze=False)
        axes = axes.flatten()
        list(map(lambda ax: ax.axis('off'), axes))

    for i, (image, target, output) in enumerate(zip(images, targets, outputs)):
        image = (image.permute(1, 2, 0) * 255.).cpu().numpy().astype('uint8').copy()

        # draw prediction
        omasks, oscores, olabels = output['masks'], output['scores'], output['labels']
        msl = list(zip(*filter(lambda mbs: mbs[1] > 0.1, zip(omasks, oscores, olabels))))
        if msl:
            omasks, oscores, olabels = msl
        else:
            omasks, oscores, olabels = [], [], []

        for score, mask, lbl in zip(oscores, omasks, olabels):
            mask = (mask[0].cpu().numpy() > 0.5).astype(np.uint8)
            # overlay_mask_with_opacity(image, mask, (255, 0, 0), 0.5)
            contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
            colour = bgr_colour_for_class(lbl)
            cv2.drawContours(image, contours, -1, colour, -1)

        # draw ground truth
        for mask, lbl in zip(target['masks'], target['labels']):
            mask = mask.cpu().numpy()
            contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
            colour = bgr_colour_for_class(int(lbl))
            cv2.drawContours(image, contours, -1, (0, 0, 0), 2)
            cv2.drawContours(image, contours, -1, colour, 1)

        cv2.imwrite(f'{output_dir}/{prefix}seg_{i}_epoch={epoch}.jpg', image)

        if should_show_visualisations:
            axes[i].imshow(image[..., ::-1])

    if should_show_visualisations:
        plt.tight_layout()
        plt.show()
        plt.close()
