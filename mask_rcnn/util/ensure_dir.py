from pathlib import Path


def ensure_dir(dirname: str) -> str:
    Path(dirname).mkdir(parents=True, exist_ok=True)
    return dirname
