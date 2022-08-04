from .config import get_config, finalise

from .actions import build_action


def run_experiment(config_file_path: str):
    cfg = get_config()
    cfg.merge_from_file(config_file_path)
    finalise(cfg)

    with build_action(cfg) as action:
        action.act()
