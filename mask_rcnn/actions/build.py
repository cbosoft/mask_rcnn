from ..config import CfgNode
from .action_base import Action
from .trainer import Trainer


ACTIONS_BY_NAME = dict(
    train=Trainer
)


def build_action(cfg: CfgNode) -> Action:
    action = ACTIONS_BY_NAME[cfg.action](cfg)
    return action
