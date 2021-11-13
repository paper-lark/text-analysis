# -*- coding: utf-8 -*-
import os
from pathlib import Path


def create_dir(dir_path: Path):
    dir_path.mkdir(parents=True, exist_ok=True, mode=0o755)
