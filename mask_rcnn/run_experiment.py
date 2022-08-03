from .config import get_config, finalise

from .actions.trainer import Trainer


def run_experiment(config_file_path: str):
    cfg = get_config()
    cfg.merge_from_file(config_file_path)
    finalise(cfg)

    trainer = Trainer(cfg)
    trainer.train()
