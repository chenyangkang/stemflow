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

from ..utils.generate_soft_colors import generate_soft_color
from ..utils.validation import check_random_state
from .Q_blocks import Grid, Node, Point

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

warnings.filterwarnings("ignore")


class QuadGrid:
    """A QuadGrid class (fixed gird length binning)"""

    def __init__(
        self,
        grid_len: Union[float, int],
        points_lower_threshold: int,
        lon_lat_equal_grid: bool = True,
        rotation_angle: Union[float, int] = 0,
        calibration_point_x_jitter: Union[float, int] = 0,
        calibration_point_y_jitter: Union[float, int] = 0,
    ):
        self.points_lower_threshold = points_lower_threshold
        self.grid_len = grid_len
        self.lon_lat_equal_grid = lon_lat_equal_grid  # Not used anyway
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
        """For completeness"""
        pass

    def get_points(self):
        """For completeness"""
        pass

    def subdivide(self):
        """Called subdivide, but actually iterative divide"""

        # Generate grids
        x_list = np.array([i.x for i in self.points])
        y_list = np.array([i.y for i in self.points])
        xmin = np.min(x_list)
        xmax = np.max(x_list)
        ymin = np.min(y_list)
        ymax = np.max(y_list)

        self.x_start = xmin - self.grid_len + self.calibration_point_x_jitter
        self.x_end = xmax + self.grid_len + self.calibration_point_x_jitter
        self.y_start = ymin - self.grid_len + self.calibration_point_y_jitter
        self.y_end = ymax + self.grid_len + self.calibration_point_y_jitter
        x_grids = np.arange(self.x_start, self.x_end, self.grid_len)
        y_grids = np.arange(self.y_start, self.y_end, self.grid_len)

        # Save the grid nodes
        self.node_list = []

        # Create grid objects for each combination of x and y ranges
        self.grids = []
        for i in range(len(x_grids) - 1):
            for j in range(len(y_grids) - 1):
                gird = Grid(i, j, (x_grids[i], x_grids[i + 1]), (y_grids[j], y_grids[j + 1]))
                self.grids.append(gird)

        # Use numpy.digitize to bin points into grids
        x_bins = np.digitize(x_list, x_grids) - 1
        y_bins = np.digitize(y_list, y_grids) - 1

        # Assign points to the corresponding grids
        for grid in self.grids:
            indices = np.where((x_bins == grid.x_index) & (y_bins == grid.y_index))[0]
            grid.points = [self.points[i] for i in indices]

    def graph(self, scatter: bool = True, ax=None):
        the_color = generate_soft_color()

        theta = -(self.rotation_angle / 360) * np.pi * 2
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        for grid in self.grids:
            xy0_trans = np.array([[grid.x_range[0], grid.y_range[0]]])
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
                        (new_x, new_y),
                        self.grid_len,
                        self.grid_len,
                        fill=False,
                        angle=self.rotation_angle,
                        color=the_color,
                    )
                )
            else:
                ax.add_patch(
                    patches.Rectangle(
                        (new_x, new_y),
                        self.grid_len,
                        self.grid_len,
                        fill=False,
                        angle=self.rotation_angle,
                        color=the_color,
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

        point_indexes_list = []
        point_grid_width_list = []
        point_grid_height_list = []
        point_grid_points_number_list = []
        calibration_point_list = []
        for grid in self.grids:
            point_indexes_list.append([point.index for point in grid.points])
            point_grid_width_list.append(self.grid_len)
            point_grid_height_list.append(self.grid_len)
            point_grid_points_number_list.append(len(grid.points))
            calibration_point_list.append((round(grid.x_range[0], 6), round(grid.x_range[0], 6)))

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
