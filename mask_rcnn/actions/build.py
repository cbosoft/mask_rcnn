from ..config import CfgNode
from .action_base import Action
from .trainer import Trainer
from .xval import CrossValidator
from .contrastive import ContrastiveTrainer


ACTIONS_BY_NAME = dict(
    train=Trainer,
    xval=CrossValidator,
    contrastive=ContrastiveTrainer,
)


def build_action(cfg: CfgNode) -> Action:
    action = ACTIONS_BY_NAME[cfg.action](cfg)
    return action
