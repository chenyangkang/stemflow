import os
import warnings
from collections.abc import Sequence

# from multiprocessing import Pool
from typing import Tuple, Union

import matplotlib.patches as patches
import matplotlib.pyplot as plt  # plotting libraries
import numpy as np
import pandas
import pandas as pd

from .generate_soft_colors import generate_soft_color
from .validation import check_random_state

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

warnings.filterwarnings("ignore")


class QuadGrid:
    """A QuadTree class"""

    def __init__(
        self,
        grid_len_lon_upper_threshold: Union[float, int],
        grid_len_lon_lower_threshold: Union[float, int],
        grid_len_lat_upper_threshold: Union[float, int],
        grid_len_lat_lower_threshold: Union[float, int],
        points_lower_threshold: int,
        lon_lat_equal_grid: bool = True,
        rotation_angle: Union[float, int] = 0,
        calibration_point_x_jitter: Union[float, int] = 0,
        calibration_point_y_jitter: Union[float, int] = 0,
    ):
        raise NotImplementedError("QuadGrid for STEM not available yet")
