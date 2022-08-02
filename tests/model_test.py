from mask_rcnn.config import get_config, finalise
from mask_rcnn.model.build import build_model
from mask_rcnn.dataset.coco import COCODataset


def test_model():
    cfg = get_config()
    cfg.data.pattern = []
    finalise(cfg)
    model = build_model(cfg)
    _ = model


def test_model_and_dataset():
    cfg = get_config()
    cfg.data.pattern = '../data/MISC_DEV_TEST_DATA.json'
    cfg.model.n_classes = 3
    finalise(cfg)
    model = build_model(cfg)
    ds = COCODataset.from_config(cfg)

    model.train()
    batch = ds[0]

    a = model([batch['image']], [batch['target']])
