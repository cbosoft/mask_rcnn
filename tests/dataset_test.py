from mask_rcnn.config import get_config, finalise
from mask_rcnn.dataset.coco import COCO_Image, COCODataset


def test_coco_dataset():
    cfg = get_config()
    cfg.data.pattern = '../data/RDavis_KickOff.json'
    finalise(cfg)
    ds = COCODataset.from_config(cfg)
