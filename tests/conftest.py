from typing import Union
from PIL import Image
import torch
import numpy as np
import pytest
from pathlib import Path
from einops import rearrange
import shutil

def pytest_sessionstart(session):
    output_path = Path(__file__).parent / "output"
    if output_path.exists():
        shutil.rmtree(output_path)
