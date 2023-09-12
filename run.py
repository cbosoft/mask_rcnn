#!/usr/bin/env python
import argparse
import os

from email_notifier import EmailNotifier

from mask_rcnn.run import run_experiment, run_template


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', nargs='+')
    parser.add_argument('--no-tmux', '-n', action='store_true', default=False)
    return parser.parse_args()


class ExperimentFailed(Exception):
    pass


if __name__ == '__main__':
    args = parse_args()

    if not args.no_tmux:
        assert os.getenv('TMUX') is not None, 'Running outside of tmux session! pass "--no-tmux" to allow this.'

    s = 's' if len(args.experiment) > 1 else ''
    with EmailNotifier(message=f'Mask R-CNN experiment{s} complete', error_message=f'Error running Mask-RCNN experiment{s}', is_html=True) as em:
        for expt in args.experiment:
            run_func = run_template if expt.endswith('.syt') else run_experiment
            try:
                run_func(expt)
            except Exception as e:
                raise ExperimentFailed(f'Error running experiment "{expt}"') from e
            except KeyboardInterrupt:
                exit(0)
            em.send_message(
                f'Mask R-CNN experiment "{expt}" complete',
                em.subject.format(kind='Update'),
                is_html=True,
                **em.config
            )
