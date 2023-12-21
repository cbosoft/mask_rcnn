from datetime import datetime, timedelta

from tqdm import tqdm
from tqdm.notebook import tqdm as notebook_tqdm

from .util import fmt_time



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
        self.total = total or len(it) or 1
        self.i = -1
        self.description = desc
        self.unit = unit
        if self.unit:
            self.unit = ' ' + self.unit
        self.time_last_display = datetime.fromtimestamp(0)
        self.start_time = None

    def __iter__(self):
        self.start_time = datetime.now()
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

    def est_ttg(self, now) -> int:
        delta: timedelta = now - self.start_time
        n_done = self.i
        n_to_go = self.total - self.i - self.initial
        t_to_go = delta.total_seconds() * n_to_go / n_done
        return int(t_to_go)

    def display(self, force=False):
        now = datetime.now()
        delta = now - self.time_last_display
        if force or (delta > timedelta(milliseconds=500)):
            progress = self.i + self.initial
            perc = int(progress * 100 / self.total)
            now_f = now.strftime('%Y-%m-%d %H:%M:%S')
            t_to_go = ''
            if self.i:
                t_to_go = self.est_ttg(now)
                t_to_go = fmt_time(t_to_go)
                t_to_go = f'({t_to_go} to go)'
            print(f'[{now_f}] {progress}/{self.total}{self.unit} ({perc}%) {self.description} {t_to_go}')
            self.time_last_display = now


def progressbar(*args, **kwargs):
    if WORKING_ENVIRONMENT == 'notebook':
        return notebook_tqdm(*args, **kwargs)
    if WORKING_ENVIRONMENT == 'detached':
        return DetachedBar(*args, **kwargs)
    else:
        return tqdm(*args, **kwargs)
