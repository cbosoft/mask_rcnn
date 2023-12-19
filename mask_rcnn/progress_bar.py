from datetime import datetime, timedelta

from tqdm import tqdm
from tqdm.notebook import tqdm as notebook_tqdm



WORKING_ENVIRONMENT = 'cli'  #  or 'notebook' or 'detached'


def _set_environment(env_kind: str):
    global WORKING_ENVIRONMENT
    WORKING_ENVIRONMENT = env_kind


def set_is_notebook():
    _set_environment('notebook')
    
def set_is_cli():
    _set_environment('cli')
    
def set_is_detached():
    _set_environment('detached')


class DetachedBar:

    def __init__(self, it, *args, initial=None, total=None, unit='', desc='', **kwargs):
        self.iter = iter(it)
        self.initial = initial or 0
        self.total = total or len(it)
        self.i = 0
        self.description = desc
        self.unit = unit
        if self.unit:
            self.unit = ' ' + self.unit
        self.time_last_display = datetime.now()

    def __iter__(self):
        self.display(True)
        return self

    def __next__(self):
        self.update()
        try:
            return next(self.iter)
        except StopIteration:
            self.display(True)
            raise

    def set_description(self, desc: str, display_now=True):
        self.description = desc
        if display_now:
            self.display()

    def update(self, n=1):
        self.i += n
        for i in range(n-1):
            next(self.iter)
        self.display()

    def display(self, force=False):
        now = datetime.now()
        if force or ((self.time_last_display - now) > timedelta(milliseconds=500)):
            progress = self.i + self.initial
            perc = int(progress * 100 / self.total)
            now_f = now.strftime('%Y-%m-%d %H:%M:%S')
            print(f'[{now_f}] {progress}/{self.total}{self.unit} ({perc}%) {self.description}')
            self.time_last_display = now


def progressbar(*args, **kwargs):
    if WORKING_ENVIRONMENT == 'notebook':
        return notebook_tqdm(*args, **kwargs)
    if WORKING_ENVIRONMENT == 'detached':
        return DetachedBar(*args, **kwargs)
    else:
        return tqdm(*args, **kwargs)
