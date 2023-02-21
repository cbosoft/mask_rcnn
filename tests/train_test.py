from mldb.config import CONFIG

from mask_rcnn.config import get_config, finalise
from mask_rcnn.actions.trainer import Trainer


def test_trainer():
    cfg = get_config()
    cfg.data.pattern = '../data/MISC_DEV_TEST_DATA.json'
    cfg.model.n_classes = 4
    cfg.training.n_epochs = 1
    finalise(cfg)
    with Trainer(cfg) as trainer:
        trainer.train()
