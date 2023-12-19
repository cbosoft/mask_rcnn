import os
from datetime import datetime

from mask_rcnn.run import run
from mask_rcnn.progress_bar import set_is_detached



if __name__ == '__main__':
    set_is_detached()
    experiment = os.getenv("EXPERIMENT")
    assert experiment is not None, 'EXPERIMENT env var must be set and be a valid path to an experiment config or template.'
    assert os.path.exists(experiment), 'EXPERIMENT must be a path to a config or template to run.'
    assert not experiment.endswith('syt'), 'template not yet supported'

    jobname = os.getenv("JOBNAME")
    assert jobname is not None, 'JOBNAME evironment variable must be set.'
    assert jobname == datetime.now().strftime(jobname), 'JOBNAME should not contain any datetime format specifiers (you want it to be constant)'

    results = f'training_results/{jobname}'
    config = f'{results}/config.yaml'
    if os.path.exists(config):
        print('Using previous job config')
        experiment = config

    run(experiment, f'training_results/{jobname}')

