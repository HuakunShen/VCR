import os
import shutil
from typing import Union
from pathlib2 import Path


def clear_dir(path: Union[str, Path]) -> None:
    """
    Helper method to clear a given directory without removing the target directory, only its children
    :param path: path of directory to clear
    :return: None
    """
    path = Path(path) if isinstance(path, str) else path
    if path.exists():
        for p in path.iterdir():
            os.remove(str(p)) if p.is_file() else shutil.rmtree(str(p))
