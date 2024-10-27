"A function module to get quadtree results for spherical indexing system. Twins to `quadtree.py`, Returns ensemble_df and plotting axes."

import os
import warnings
from functools import partial
from typing import Tuple, Union

import joblib
import matplotlib
import matplotlib.pyplot as plt  # plotting libraries
import numpy as np
import pandas
import pandas as pd
from tqdm import tqdm

from ..gridding.Sphere_QTree import Sphere_QTree
from .quadtree import generate_temporal_bins
from .sphere.coordinate_transform import lonlat_cartesian_3D_transformer
from .validation import check_random_state

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

warnings.filterwarnings("ignore")


def get_one_ensemble_sphere_quadtree(
    ensemble_count,
    data: pandas.core.frame.DataFrame,
    Temporal1: str = "DOY",
    grid_len_upper_threshold: Union[float, int] = 8000,
    grid_len_lower_threshold: Union[float, int] = 500,
    points_lower_threshold: int = 50,
    temporal_start: Union[float, int] = 1,
    temporal_end: Union[float, int] = 366,
    temporal_step: Union[float, int] = 20,
    temporal_bin_interval: Union[float, int] = 50,
    temporal_bin_start_jitter: Union[float, int, str] = "adaptive",
    spatio_bin_jitter_magnitude: Union[float, int] = "adaptive",
    save_gridding_plotly: bool = True,
    save_gridding_plot: bool = False,
    ax=None,
    radius: Union[int, float] = 6371.0,
    plot_empty: bool = False,
    rng: np.random._generator.Generator = None,
):
    """Generate QuadTree gridding based on the input dataframe
    A function to get quadtree results for spherical indexing system. Twins to `get_ensemble_quadtree` in `quadtree.py`, Returns ensemble_df and plotting axes.

    Args:
        data:
            Input pandas-like dataframe
        Temporal1:
            Temporal column name 1 in data
        size:
            How many ensemble to generate (how many round the data are gone through)
        grid_len_upper_threshold:
            force divide if grid longitude larger than the threshold (in km)
        grid_len_lower_threshold:
            stop divide if grid longitude **will** be below than the threshold (in km)
        points_lower_threshold:
            Do not train the model if the available data records for this stixel is less than this threshold,
            and directly set the value to np.nan.
        temporal_start:
            start of the temporal sequence
        temporal_end:
            end of the temporal sequence
        temporal_step:
            step of the sliding window
        temporal_bin_interval:
            size of the sliding window
        temporal_bin_start_jitter:
            jitter of the start of the sliding window.
            If 'adaptive', a adaptive jitter of range (-bin_interval, 0) will be generated
            for the start.
        spatio_bin_jitter_magnitude:
            jitter of the spatial gridding.
        save_gridding_plotly:
            Whether to save the plotly interactive gridding plot.
        save_gridding_plot:
            Whether ot save gridding plots
        ax:
            Matplotlib Axes to add to.
        radius (Union[int, float]):
            The radius of earth in km. Defaults to 6371.0.
        rng:
            random number generator.

    Returns:
        A tuple of <br>
            1. ensemble dataframe;<br>
            2. grid plot. np.nan if save_gridding_plot=False<br>

    """
    rng = check_random_state(rng)

    if spatio_bin_jitter_magnitude == "adaptive":
        rotation_angle = rng.uniform(0, 90)
        rotation_axis = rng.uniform(-1, 1, 3)

    temporal_bins = generate_temporal_bins(
        start=temporal_start,
        end=temporal_end,
        step=temporal_step,
        bin_interval=temporal_bin_interval,
        temporal_bin_start_jitter=temporal_bin_start_jitter,
        rng=rng,
    )

    ensemble_all_df_list = []

    for time_block_index, bin_ in enumerate(temporal_bins):
        time_start = bin_[0]
        time_end = bin_[1]
        sub_data = data[(data[Temporal1] >= time_start) & (data[Temporal1] < time_end)].copy()
        if len(sub_data) == 0:
            continue

        # Transform lon lat to 3D cartesian
        x, y, z = lonlat_cartesian_3D_transformer.transform(sub_data["longitude"], sub_data["latitude"], radius=radius)
        sub_data.loc[:, "x_3D"] = x
        sub_data.loc[:, "y_3D"] = y
        sub_data.loc[:, "z_3D"] = z

        QT_obj = Sphere_QTree(
            grid_len_upper_threshold=grid_len_upper_threshold,
            grid_len_lower_threshold=grid_len_lower_threshold,
            points_lower_threshold=points_lower_threshold,
            rotation_angle=rotation_angle,
            rotation_axis=rotation_axis,
            radius=radius,
            plot_empty=plot_empty,
        )

        # Give the data and indexes. The indexes should be used to assign points data so that base model can run on those points,
        # You need to generate the splitting parameters once giving the data. Like the calibration point and min,max.
        QT_obj.add_3D_data(sub_data.index, sub_data["x_3D"].values, sub_data["y_3D"].values, sub_data["z_3D"].values)
        QT_obj.generate_gridding_params()

        # Call subdivide to precess
        QT_obj.subdivide()
        this_slice = QT_obj.get_final_result()

        if save_gridding_plot:
            if time_block_index == int(len(temporal_bins) / 2):
                QT_obj.graph(scatter=False, ax=ax)

        if save_gridding_plotly:
            if time_block_index == int(len(temporal_bins) / 2):
                ax = QT_obj.plotly_graph(scatter=False, ax=ax)

        this_slice["ensemble_index"] = ensemble_count
        this_slice[f"{Temporal1}_start"] = time_start
        this_slice[f"{Temporal1}_end"] = time_end
        this_slice[f"{Temporal1}_start"] = round(this_slice[f"{Temporal1}_start"], 1)
        this_slice[f"{Temporal1}_end"] = round(this_slice[f"{Temporal1}_end"], 1)
        this_slice["unique_stixel_id"] = [
            str(time_block_index) + "_" + str(i) + "_" + str(k)
            for i, k in zip(this_slice["ensemble_index"].values, this_slice["stixel_indexes"].values)
        ]
        ensemble_all_df_list.append(this_slice)

    this_ensemble_df = pd.concat(ensemble_all_df_list).reset_index(drop=True)
    this_ensemble_df = this_ensemble_df.reset_index(drop=True)
    return this_ensemble_df
