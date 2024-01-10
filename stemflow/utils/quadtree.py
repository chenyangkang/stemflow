# import libraries
import os
import warnings
from collections.abc import Sequence
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from typing import Tuple, Union

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt  # plotting libraries
import numpy as np
import pandas
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from .generate_soft_colors import generate_soft_color
from .validation import check_random_state

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)


class Point:
    """A Point class for recording data points"""

    def __init__(self, index, x, y):
        self.x = x
        self.y = y
        self.index = index


class Node:
    """A tree-like division node class"""

    def __init__(
        self, x0: Union[float, int], y0: Union[float, int], w: Union[float, int], h: Union[float, int], points: Point
    ):
        self.x0 = x0
        self.y0 = y0
        self.width = w
        self.height = h
        self.points = points
        self.children = []

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def get_points(self):
        return self.points


def recursive_subdivide(
    node: Node,
    grid_len_lon_upper_threshold: Union[float, int],
    grid_len_lon_lower_threshold: Union[float, int],
    grid_len_lat_upper_threshold: Union[float, int],
    grid_len_lat_lower_threshold: Union[float, int],
    points_lower_threshold: Union[float, int],
):
    """recursively subdivide the grids

    Args:
        node:
            node input
        grid_len_lon_upper_threshold:
            force divide if grid longitude larger than the threshold
        grid_len_lon_lower_threshold:
            stop divide if grid longitude **will** be below than the threshold
        grid_len_lat_upper_threshold:
            force divide if grid latitude larger than the threshold
        grid_len_lat_lower_threshold:
            stop divide if grid latitude **will** be below than the threshold

    """

    if (node.width / 2 < grid_len_lon_lower_threshold) or (node.height / 2 < grid_len_lat_lower_threshold):
        return

    w_ = float(node.width / 2)
    h_ = float(node.height / 2)

    p = contains(node.x0, node.y0, w_, h_, node.points)
    x1 = Node(node.x0, node.y0, w_, h_, p)
    recursive_subdivide(
        x1,
        grid_len_lon_upper_threshold,
        grid_len_lon_lower_threshold,
        grid_len_lat_upper_threshold,
        grid_len_lat_lower_threshold,
        points_lower_threshold,
    )

    p = contains(node.x0, node.y0 + h_, w_, h_, node.points)
    x2 = Node(node.x0, node.y0 + h_, w_, h_, p)
    recursive_subdivide(
        x2,
        grid_len_lon_upper_threshold,
        grid_len_lon_lower_threshold,
        grid_len_lat_upper_threshold,
        grid_len_lat_lower_threshold,
        points_lower_threshold,
    )

    p = contains(node.x0 + w_, node.y0, w_, h_, node.points)
    x3 = Node(node.x0 + w_, node.y0, w_, h_, p)
    recursive_subdivide(
        x3,
        grid_len_lon_upper_threshold,
        grid_len_lon_lower_threshold,
        grid_len_lat_upper_threshold,
        grid_len_lat_lower_threshold,
        points_lower_threshold,
    )

    p = contains(node.x0 + w_, node.y0 + h_, w_, h_, node.points)
    x4 = Node(node.x0 + w_, node.y0 + h_, w_, h_, p)
    recursive_subdivide(
        x4,
        grid_len_lon_upper_threshold,
        grid_len_lon_lower_threshold,
        grid_len_lat_upper_threshold,
        grid_len_lat_lower_threshold,
        points_lower_threshold,
    )

    for ch_node in [x1, x2, x3, x4]:
        if len(ch_node.points) <= points_lower_threshold:
            if not ((node.width > grid_len_lon_upper_threshold) or (node.height > grid_len_lat_upper_threshold)):
                return

    node.children = [x1, x2, x3, x4]


def contains(x, y, w, h, points):
    """return list of points within the grid"""
    pts = []
    for point in points:
        if point.x >= x and point.x <= x + w and point.y >= y and point.y <= y + h:
            pts.append(point)
    return pts


def find_children(node):
    """return children nodes of this node"""
    if not node.children:
        return [node]
    else:
        children = []
        for child in node.children:
            children += find_children(child)
    return children


class QTree:
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
        """Create a QuadTree object

        Args:
            grid_len_lon_upper_threshold:
                force divide if grid longitude larger than the threshold
            grid_len_lon_lower_threshold:
                stop divide if grid longitude **will** be below than the threshold
            grid_len_lat_upper_threshold:
                force divide if grid latitude larger than the threshold
            grid_len_lat_lower_threshold:
                stop divide if grid latitude **will** be below than the threshold
            points_lower_threshold:
                stop divide if points count is less than this threshold.
            lon_lat_equal_grid:
                whether to split the longitude and latitude equally.
            rotation_angle:
                angles to rotate the gridding.
            calibration_point_x_jitter:
                jittering the gridding on longitude.
            calibration_point_y_jitter:
                jittering the gridding on latitude.

        Example:
            ```py
            >> QT_obj = QTree(grid_len_lon_upper_threshold=25,
                            grid_len_lon_lower_threshold=5,
                            grid_len_lat_upper_threshold=25,
                            grid_len_lat_lower_threshold=5,
                            points_lower_threshold=50,
                            lon_lat_equal_grid = True,
                            rotation_angle = 15.5,
                            calibration_point_x_jitter = 10,
                            calibration_point_y_jitter = 10)
            >> QT_obj.add_lon_lat_data(sub_data.index, sub_data['longitude'].values, sub_data['latitude'].values)
            >> QT_obj.generate_gridding_params()
            >> QT_obj.subdivide() # Call subdivide to process
            >> gridding_info = QT_obj.get_final_result()  # gridding_info is a dataframe
            ```

        """

        self.points_lower_threshold = points_lower_threshold
        self.grid_len_lon_upper_threshold = grid_len_lon_upper_threshold
        self.grid_len_lon_lower_threshold = grid_len_lon_lower_threshold
        self.grid_len_lat_upper_threshold = grid_len_lat_upper_threshold
        self.grid_len_lat_lower_threshold = grid_len_lat_lower_threshold
        self.lon_lat_equal_grid = lon_lat_equal_grid
        # self.points = [Point(random.uniform(0, 10), random.uniform(0, 10)) for x in range(n)]
        self.points = []
        self.rotation_angle = rotation_angle
        self.calibration_point_x_jitter = calibration_point_x_jitter
        self.calibration_point_y_jitter = calibration_point_y_jitter

    def add_lon_lat_data(self, indexes: Sequence, x_array: Sequence, y_array: Sequence):
        """Store input lng lat data and transform to **Point** object

        Parameters:
            indexes: Unique identifier for indexing the point.
            x_array: longitudinal values.
            y_array: latitudinal values.

        """
        if not len(x_array) == len(y_array) or not len(x_array) == len(indexes):
            raise ValueError("input longitude and latitude and indexes not in same length!")

        data = np.array([x_array, y_array]).T
        angle = self.rotation_angle
        r = angle / 360
        theta = r * np.pi * 2
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        data = data @ rotation_matrix
        lon_new = (data[:, 0] + self.calibration_point_x_jitter).tolist()
        lat_new = (data[:, 1] + self.calibration_point_y_jitter).tolist()

        for index, lon, lat in zip(indexes, lon_new, lat_new):
            self.points.append(Point(index, lon, lat))

    def generate_gridding_params(self):
        """generate the gridding params after data are added

        Raises:
            ValueError: self.lon_lat_equal_grid is not a bool

        """
        x_list = [i.x for i in self.points]
        y_list = [i.y for i in self.points]
        self.grid_length_x = np.max(x_list) - np.min(x_list)
        self.grid_length_y = np.max(y_list) - np.min(y_list)

        left_bottom_point_x = np.min(x_list)
        left_bottom_point_y = np.min(y_list)

        self.left_bottom_point = (left_bottom_point_x, left_bottom_point_y)
        if self.lon_lat_equal_grid is True:
            self.root = Node(
                left_bottom_point_x,
                left_bottom_point_y,
                max(self.grid_length_x, self.grid_length_y),
                max(self.grid_length_x, self.grid_length_y),
                self.points,
            )
        elif self.lon_lat_equal_grid is False:
            self.root = Node(
                left_bottom_point_x, left_bottom_point_y, self.grid_length_x, self.grid_length_y, self.points
            )
        else:
            raise ValueError("The input lon_lat_equal_grid not a boolean value!")

    def get_points(self):
        """return points"""
        return self.points

    def subdivide(self):
        """start recursively subdivide"""
        recursive_subdivide(
            self.root,
            self.grid_len_lon_upper_threshold,
            self.grid_len_lon_lower_threshold,
            self.grid_len_lat_upper_threshold,
            self.grid_len_lat_lower_threshold,
            self.points_lower_threshold,
        )

    def graph(self, scatter: bool = True, ax=None):
        """plot gridding

        Args:
            scatter: Whether add scatterplot of data points
        """
        the_color = generate_soft_color()

        c = find_children(self.root)

        areas = set()
        width_set = set()
        height_set = set()
        for el in c:
            areas.add(el.width * el.height)
            width_set.add(el.width)
            height_set.add(el.height)

        theta = -(self.rotation_angle / 360) * np.pi * 2
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        for n in c:
            xy0_trans = np.array([[n.x0, n.y0]])
            if self.calibration_point_x_jitter:
                new_x = xy0_trans[:, 0] - self.calibration_point_x_jitter
            else:
                new_x = xy0_trans[:, 0]

            if self.calibration_point_y_jitter:
                new_y = xy0_trans[:, 1] - self.calibration_point_y_jitter
            else:
                new_y = xy0_trans[:, 1]
            new_xy = np.array([[new_x[0], new_y[0]]]) @ rotation_matrix
            new_x = new_xy[:, 0]
            new_y = new_xy[:, 1]

            if ax is None:
                plt.gcf().gca().add_patch(
                    patches.Rectangle(
                        (new_x, new_y), n.width, n.height, fill=False, angle=self.rotation_angle, color=the_color
                    )
                )
            else:
                ax.add_patch(
                    patches.Rectangle(
                        (new_x, new_y), n.width, n.height, fill=False, angle=self.rotation_angle, color=the_color
                    )
                )

        x = np.array([point.x for point in self.points]) - self.calibration_point_x_jitter
        y = np.array([point.y for point in self.points]) - self.calibration_point_y_jitter

        data = np.array([x, y]).T @ rotation_matrix
        if scatter:
            if ax is None:
                plt.scatter(
                    data[:, 0].tolist(), data[:, 1].tolist(), s=0.2, c="tab:blue", alpha=0.7
                )  # plots the points as red dots
            else:
                ax.scatter(
                    data[:, 0].tolist(), data[:, 1].tolist(), s=0.2, c="tab:blue", alpha=0.7
                )  # plots the points as red dots
        return

    def get_final_result(self) -> pandas.core.frame.DataFrame:
        """get points assignment to each grid and transform the data into pandas df.

        Returns:
            results (DataFrame): A pandas dataframe containing the gridding information
        """
        all_grids = find_children(self.root)
        point_indexes_list = []
        point_grid_width_list = []
        point_grid_height_list = []
        point_grid_points_number_list = []
        calibration_point_list = []
        for grid in all_grids:
            point_indexes_list.append([point.index for point in grid.points])
            point_grid_width_list.append(grid.width)
            point_grid_height_list.append(grid.height)
            point_grid_points_number_list.append(len(grid.points))
            calibration_point_list.append((round(grid.x0, 6), round(grid.y0, 6)))

        result = pd.DataFrame(
            {
                "checklist_indexes": point_indexes_list,
                "stixel_indexes": list(range(len(point_grid_width_list))),
                "stixel_width": point_grid_width_list,
                "stixel_height": point_grid_height_list,
                "stixel_checklist_count": point_grid_points_number_list,
                "stixel_calibration_point(transformed)": calibration_point_list,
                "rotation": [self.rotation_angle] * len(point_grid_width_list),
                "space_jitter(first rotate by zero then add this)": [
                    (round(self.calibration_point_x_jitter, 6), round(self.calibration_point_y_jitter, 6))
                ]
                * len(point_grid_width_list),
            }
        )

        result = result[result["stixel_checklist_count"] != 0]
        return result


def generate_temporal_bins(
    start: Union[float, int],
    end: Union[float, int],
    step: Union[float, int],
    bin_interval: Union[float, int],
    temporal_bin_start_jitter: Union[float, int, str] = "random",
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
            If 'random', a random jitter of range (-bin_interval, 0) will be generated
            for the start.

    Returns:
        A list of tuple. Start and end of each temporal bin.

    """
    bin_interval = bin_interval  # 50
    step = step  # 20

    if type(temporal_bin_start_jitter) == str and temporal_bin_start_jitter == "random":
        jit = np.random.uniform(low=0, high=bin_interval)
    elif type(temporal_bin_start_jitter) in [int, float]:
        jit = temporal_bin_start_jitter

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


def generate_one_ensemble(
    ensemble_count,
    spatio_bin_jitter_magnitude,
    temporal_start,
    temporal_end,
    temporal_step,
    temporal_bin_interval,
    temporal_bin_start_jitter,
    data,
    Temporal1,
    grid_len_lon_upper_threshold,
    grid_len_lon_lower_threshold,
    grid_len_lat_upper_threshold,
    grid_len_lat_lower_threshold,
    points_lower_threshold,
    Spatio1,
    Spatio2,
    save_gridding_plot,
    ax,
):
    this_ensemble = []
    rotation_angle = np.random.uniform(0, 360)
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
        this_ensemble.append(this_slice)

    return pd.concat(this_ensemble, axis=0)


def get_ensemble_quadtree(
    data: pandas.core.frame.DataFrame,
    Spatio1: str = "longitude",
    Spatio2: str = "latitude",
    Temporal1: str = "DOY",
    size: str = 1,
    grid_len_lon_upper_threshold: Union[float, int] = 25,
    grid_len_lon_lower_threshold: Union[float, int] = 5,
    grid_len_lat_upper_threshold: Union[float, int] = 25,
    grid_len_lat_lower_threshold: Union[float, int] = 5,
    points_lower_threshold: int = 50,
    temporal_start: Union[float, int] = 1,
    temporal_end: Union[float, int] = 366,
    temporal_step: Union[float, int] = 20,
    temporal_bin_interval: Union[float, int] = 50,
    temporal_bin_start_jitter: Union[float, int, str] = "random",
    spatio_bin_jitter_magnitude: Union[float, int] = 10,
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
            If 'random', a random jitter of range (-bin_interval, 0) will be generated
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
        partial_generate_one_ensemble = partial(
            generate_one_ensemble,
            spatio_bin_jitter_magnitude=spatio_bin_jitter_magnitude,
            temporal_start=temporal_start,
            temporal_end=temporal_end,
            temporal_step=temporal_step,
            temporal_bin_interval=temporal_bin_interval,
            temporal_bin_start_jitter=temporal_bin_start_jitter,
            data=data,
            Temporal1=Temporal1,
            grid_len_lon_upper_threshold=grid_len_lon_upper_threshold,
            grid_len_lon_lower_threshold=grid_len_lon_lower_threshold,
            grid_len_lat_upper_threshold=grid_len_lat_upper_threshold,
            grid_len_lat_lower_threshold=grid_len_lat_lower_threshold,
            points_lower_threshold=points_lower_threshold,
            Spatio1=Spatio1,
            Spatio2=Spatio2,
            save_gridding_plot=save_gridding_plot,
        )

        ensemble_all_df_list = process_map(partial_generate_one_ensemble, list(range(size)), max_workers=njobs)

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
