# -*- coding: utf-8 -*-
import os
from pathlib import Path


def create_dir(dir_path: Path):
    try:
        os.mkdir(dir_path, mode=0o755)
    except FileExistsError:
        pass
