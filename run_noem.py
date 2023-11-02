#!/usr/bin/env python
import argparse
import os
import mlflow

from mask_rcnn.run import run_experiment, run_template


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', nargs='+')
    parser.add_argument('--no-tmux', '-n', action='store_true', default=False)
    return parser.parse_args()


class ExperimentFailed(Exception):
    pass


mlflow.set_tracking_uri('http://130.159.94.187:5454')


if __name__ == '__main__':
    args = parse_args()

    if not args.no_tmux:
        assert os.getenv('TMUX') is not None, 'Running outside of tmux session! pass "--no-tmux" to allow this.'

    for expt in args.experiment:
        run_func = run_template if expt.endswith('.syt') else run_experiment
        run_func(expt)
