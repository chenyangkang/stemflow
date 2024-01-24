"A function module to get quadtree results for 2D indexing system. Returns ensemble_df and plotting axes."

import os
import warnings
from typing import Tuple, Union

import matplotlib
import matplotlib.pyplot as plt  # plotting libraries
import numpy as np
import pandas
import pandas as pd
from tqdm import tqdm

from ..gridding.QTree import QTree
from ..gridding.QuadGrid import QuadGrid
from .validation import check_transform_spatio_bin_jitter_magnitude, check_transform_temporal_bin_start_jitter

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

warnings.filterwarnings("ignore")


def generate_temporal_bins(
    start: Union[float, int],
    end: Union[float, int],
    step: Union[float, int],
    bin_interval: Union[float, int],
    temporal_bin_start_jitter: Union[float, int, str] = "adaptive",
) -> list:
    """Generate random temporal bins that splits the data

    Args:
        start:
            start of the temporal sequence
        end:
            end of the temporal sequence
        step:
            step of the sliding window
        bin_interval:
            size of the sliding window
        temporal_bin_start_jitter:
            jitter of the start of the sliding window.
            If 'adaptive', a random jitter of range (-bin_interval, 0) will be generated
            for the start.

    Returns:
        A list of tuple. Start and end of each temporal bin.

    """
    bin_interval = bin_interval  # 50
    step = step  # 20

    jit = check_transform_temporal_bin_start_jitter(temporal_bin_start_jitter, bin_interval)

    start = start - jit
    bin_list = []

    i = 0
    while True:
        s = start + i * step
        e = s + bin_interval
        if s >= end:
            break
        bin_list.append((s, e))
        i += 1

    return bin_list


def get_ensemble_quadtree(
    data: pandas.core.frame.DataFrame,
    Spatio1: str = "longitude",
    Spatio2: str = "latitude",
    Temporal1: str = "DOY",
    size: str = 1,
    grid_len: Union[None, float, int] = None,
    grid_len_lon_upper_threshold: Union[float, int] = 25,
    grid_len_lon_lower_threshold: Union[float, int] = 5,
    grid_len_lat_upper_threshold: Union[float, int] = 25,
    grid_len_lat_lower_threshold: Union[float, int] = 5,
    points_lower_threshold: int = 50,
    temporal_start: Union[float, int] = 1,
    temporal_end: Union[float, int] = 366,
    temporal_step: Union[float, int] = 20,
    temporal_bin_interval: Union[float, int] = 50,
    temporal_bin_start_jitter: Union[float, int, str] = "adaptive",
    spatio_bin_jitter_magnitude: Union[float, int] = "adaptive",
    save_gridding_plot: bool = True,
    njobs: int = 1,
    verbosity: int = 1,
    plot_xlims: Tuple[Union[float, int]] = (-180, 180),
    plot_ylims: Tuple[Union[float, int]] = (-90, 90),
    save_path: str = "",
    ax=None,
) -> Tuple[pandas.core.frame.DataFrame, Union[matplotlib.figure.Figure, float]]:
    """Generate QuadTree gridding based on the input dataframe

    Args:
        data:
            Input pandas-like dataframe
        Spatio1:
            Spatial column name 1 in data
        Spatio2:
            Spatial column name 2 in data
        Temporal1:
            Temporal column name 1 in data
        size:
            How many ensemble to generate (how many round the data are gone through)
        grid_len:
            If used by STEM, instead of AdaSTEM, the grid length will be fixed by this parameters.
            It overrides the following four gridding parameters.
        grid_len_lon_upper_threshold:
            force divide if grid longitude larger than the threshold
        grid_len_lon_lower_threshold:
            stop divide if grid longitude **will** be below than the threshold
        grid_len_lat_upper_threshold:
            force divide if grid latitude larger than the threshold
        grid_len_lat_lower_threshold:
            stop divide if grid latitude **will** be below than the threshold
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
        save_gridding_plot:
            Whether ot save gridding plots
        njobs:
            Multi-processes count.
        plot_xlims:
            If save_gridding_plot=True, what is the xlims of the plot
        plot_ylims:
            If save_gridding_plot=True, what is the ylims of the plot
        save_path:
            If not '', save the ensemble dataframe to this path
        ax:
            Matplotlib Axes to add to.

    Returns:
        A tuple of <br>
            1. ensemble dataframe;<br>
            2. grid plot. np.nan if save_gridding_plot=False<br>

    """
    spatio_bin_jitter_magnitude = check_transform_spatio_bin_jitter_magnitude(
        data, Spatio1, Spatio2, spatio_bin_jitter_magnitude
    )

    ensemble_all_df_list = []

    if save_gridding_plot:
        if ax is None:
            plt.figure(figsize=(20, 20))
            plt.xlim([plot_xlims[0], plot_xlims[1]])
            plt.ylim([plot_ylims[0], plot_ylims[1]])
            plt.title("Quadtree", fontsize=20)
        else:
            # ax.set_xlim([plot_xlims[0],plot_xlims[1]])
            pass

    if njobs > 1 and isinstance(njobs, int):
        raise NotImplementedError("Multi-threading for ensemble generation is not implemented yet.")

    else:
        iter_func_ = tqdm(range(size), total=size, desc="Generating Ensemble: ") if verbosity > 0 else range(size)

        for ensemble_count in iter_func_:
            # rotation_angle = np.random.uniform(0,90)
            rotation_angle = (90 / len(iter_func_)) * ensemble_count
            calibration_point_x_jitter = np.random.uniform(-spatio_bin_jitter_magnitude, spatio_bin_jitter_magnitude)
            calibration_point_y_jitter = np.random.uniform(-spatio_bin_jitter_magnitude, spatio_bin_jitter_magnitude)

            # print(f'ensemble_count: {ensemble_count}')
            temporal_bins = generate_temporal_bins(
                start=temporal_start,
                end=temporal_end,
                step=temporal_step,
                bin_interval=temporal_bin_interval,
                temporal_bin_start_jitter=temporal_bin_start_jitter,
            )

            for time_block_index, bin_ in enumerate(temporal_bins):
                time_start = bin_[0]
                time_end = bin_[1]
                sub_data = data[(data[Temporal1] >= time_start) & (data[Temporal1] < time_end)]

                if len(sub_data) == 0:
                    continue

                if grid_len is None:
                    QT_obj = QTree(
                        grid_len_lon_upper_threshold=grid_len_lon_upper_threshold,
                        grid_len_lon_lower_threshold=grid_len_lon_lower_threshold,
                        grid_len_lat_upper_threshold=grid_len_lat_upper_threshold,
                        grid_len_lat_lower_threshold=grid_len_lat_lower_threshold,
                        points_lower_threshold=points_lower_threshold,
                        lon_lat_equal_grid=True,
                        rotation_angle=rotation_angle,
                        calibration_point_x_jitter=calibration_point_x_jitter,
                        calibration_point_y_jitter=calibration_point_y_jitter,
                    )
                elif isinstance(grid_len, float) or isinstance(grid_len, int):
                    QT_obj = QuadGrid(
                        grid_len=grid_len,
                        points_lower_threshold=points_lower_threshold,
                        lon_lat_equal_grid=True,
                        rotation_angle=rotation_angle,
                        calibration_point_x_jitter=calibration_point_x_jitter,
                        calibration_point_y_jitter=calibration_point_y_jitter,
                    )
                else:
                    raise TypeError("grid_len passed is not int or float.")

                # Give the data and indexes. The indexes should be used to assign points data so that base model can run on those points,
                # You need to generate the splitting parameters once giving the data. Like the calibration point and min,max.

                QT_obj.add_lon_lat_data(sub_data.index, sub_data[Spatio1].values, sub_data[Spatio2].values)
                QT_obj.generate_gridding_params()

                # Call subdivide to precess
                QT_obj.subdivide()
                this_slice = QT_obj.get_final_result()

                if save_gridding_plot:
                    if time_block_index == int(len(temporal_bins) / 2):
                        QT_obj.graph(scatter=False, ax=ax)

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

    ensemble_df = pd.concat(ensemble_all_df_list).reset_index(drop=True)
    ensemble_df.loc[:, "stixel_calibration_point_transformed_left_bound"] = [
        i[0] for i in ensemble_df["stixel_calibration_point(transformed)"]
    ]
    ensemble_df.loc[:, "stixel_calibration_point_transformed_lower_bound"] = [
        i[1] for i in ensemble_df["stixel_calibration_point(transformed)"]
    ]
    ensemble_df.loc[:, "stixel_calibration_point_transformed_right_bound"] = (
        ensemble_df["stixel_calibration_point_transformed_left_bound"] + ensemble_df["stixel_width"]
    )
    ensemble_df.loc[:, "stixel_calibration_point_transformed_upper_bound"] = (
        ensemble_df["stixel_calibration_point_transformed_lower_bound"] + ensemble_df["stixel_height"]
    )
    ensemble_df = ensemble_df.reset_index(drop=True)

    del ensemble_all_df_list

    if not save_path == "":
        ensemble_df.to_csv(save_path, index=False)
        print(f"Saved! {save_path}")

    if save_gridding_plot:
        if ax is None:
            plt.tight_layout()
            plt.gca().set_aspect("equal")
            ax = plt.gcf()
            plt.close()

        else:
            pass

        return ensemble_df, ax

    else:
        return ensemble_df, np.nan
