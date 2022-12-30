from .config import CfgNode, get_config, finalise

from .actions import build_action


def run_experiment(config_file_path: str):
    cfg = get_config()
    cfg.merge_from_file(config_file_path)
    run_config(cfg)


def run_config(cfg: CfgNode):
    finalise(cfg)

    with build_action(cfg) as action:
        action.act()


def run_template(template_filename: str):
    cfg = get_config()
    variants = cfg.apply_template(template_filename)
    print(f'Running templated experiment "{template_filename}" ({len(variants)} permutations)')
    for variant in variants:
        run_config(variant)
