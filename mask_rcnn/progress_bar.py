from tqdm import tqdm
from tqdm.notebook import tqdm as notebook_tqdm


IS_NOTEBOOK = False


def set_is_notebook():
    global IS_NOTEBOOK
    IS_NOTEBOOK = True


def progressbar(*args, **kwargs):
    if IS_NOTEBOOK:
        return notebook_tqdm(*args, **kwargs)
    else:
        return tqdm(*args, **kwargs)
