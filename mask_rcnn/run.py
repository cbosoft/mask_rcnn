from .config import CfgNode, get_config, finalise

from .actions import build_action


def run_experiment(config_file_path: str, **kwargs):
    cfg = get_config()
    cfg.merge_from_file(config_file_path)
    run_config(cfg, **kwargs)


def run_config(cfg: CfgNode, specified_output_dir: str = None):
    if specified_output_dir is not None:
        cfg.output_dir = specified_output_dir
    finalise(cfg)

    with build_action(cfg) as action:
        action.act()


def run_template(template_filename: str, specified_output_dir: str = None):
    cfg = get_config()
    variants = cfg.apply_template(template_filename)
    print(f'Running templated experiment "{template_filename}" ({len(variants)} permutations)')
    for i, variant in enumerate(variants):
        run_config(variant, specified_output_dir=f'{specified_output_dir}/{i}')


def run(filename: str, specified_output_dir: str = None):
    if filename.endswith('syt'):
        run_template(filename, specified_output_dir=specified_output_dir)
    else:
        run_experiment(filename, specified_output_dir=specified_output_dir)
