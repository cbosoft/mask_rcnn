from datetime import datetime


def today():
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
