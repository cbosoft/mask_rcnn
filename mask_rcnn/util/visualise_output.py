import cv2
import numpy as np

from matplotlib.cm import get_cmap



def visualise_output(image, masks, scores, labels, colours=None, score_thresh=0.75):
    if colours is None:
        cm = get_cmap('tab10')

        def colours(i):
            return [int(c*256) for c in cm(i)[:3]]

    image = (image.squeeze().permute(1, 2, 0).detach().cpu().numpy()*255.).astype(np.uint8)
    masks = ((masks.squeeze().detach().cpu().numpy() > 0.5)*255.).astype(np.uint8)
    scores = scores.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    for mask, lbl, score in zip(masks, labels, scores):
        if score < score_thresh:
            break

        contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
        cv2.drawContours(image, contours, -1, colours(lbl), 3)
    return image
