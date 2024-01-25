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
from ..utils.jitterrotation.jitterrotator import JitterRotator
from ..utils.validation import check_random_state
from .Q_blocks import QGrid, QNode, QPoint

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
        plot_empty: bool = False,
    ):
        """Create a QuadTree object

        Args:
            grid_len:
                grid length
            points_lower_threshold:
                skip the grid if less samples are contained
            lon_lat_equal_grid:
                whether to split the longitude and latitude equally.
            rotation_angle:
                angles to rotate the gridding.
            calibration_point_x_jitter:
                jittering the gridding on longitude.
            calibration_point_y_jitter:
                jittering the gridding on latitude.
            plot_empty:
                Whether to plot the empty grid

        Example:
            ```py
            >> QT_obj = QuadGrid(grid_len=20,
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
        self.grid_len = grid_len
        self.lon_lat_equal_grid = lon_lat_equal_grid  # Not used anyway
        self.points = []
        self.rotation_angle = rotation_angle
        self.calibration_point_x_jitter = calibration_point_x_jitter
        self.calibration_point_y_jitter = calibration_point_y_jitter
        self.plot_empty = plot_empty

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
            self.points.append(QPoint(index, lon, lat))

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

        self.x_start = xmin - self.grid_len
        self.x_end = xmax + self.grid_len
        self.y_start = ymin - self.grid_len
        self.y_end = ymax + self.grid_len
        x_grids = np.arange(self.x_start, self.x_end, self.grid_len)
        y_grids = np.arange(self.y_start, self.y_end, self.grid_len)

        # Save the grid nodes
        self.node_list = []

        # Create grid objects for each combination of x and y ranges
        self.grids = []
        for i in range(len(x_grids) - 1):
            for j in range(len(y_grids) - 1):
                gird = QGrid(i, j, (x_grids[i], x_grids[i + 1]), (y_grids[j], y_grids[j + 1]))
                self.grids.append(gird)

        # Use numpy.digitize to bin points into grids
        x_bins = np.digitize(x_list, x_grids) - 1
        y_bins = np.digitize(y_list, y_grids) - 1

        # Assign points to the corresponding grids
        for grid in self.grids:
            indices = np.where((x_bins == grid.x_index) & (y_bins == grid.y_index))[0]
            grid.points = [self.points[i] for i in indices]

    def graph(self, scatter: bool = True, ax=None):
        """plot gridding

        Args:
            scatter: Whether add scatterplot of data points
        """

        the_color = generate_soft_color()

        for grid in self.grids:
            old_x, old_y = JitterRotator.inverse_jitter_rotate(
                [grid.x_range[0]],
                [grid.y_range[0]],
                self.rotation_angle,
                self.calibration_point_x_jitter,
                self.calibration_point_y_jitter,
            )

            if ax is None:
                plt.gcf().gca().add_patch(
                    patches.Rectangle(
                        (old_x, old_y),
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
                        (old_x, old_y),
                        self.grid_len,
                        self.grid_len,
                        fill=False,
                        angle=self.rotation_angle,
                        color=the_color,
                    )
                )

        if scatter:
            old_x, old_y = JitterRotator.inverse_jitter_rotate(
                [point.x for point in self.points],
                [point.y for point in self.points],
                self.rotation_angle,
                self.calibration_point_x_jitter,
                self.calibration_point_y_jitter,
            )

            if ax is None:
                plt.scatter(old_x, old_y, s=0.2, c="tab:blue", alpha=0.7)  # plots the points as red dots
            else:
                ax.scatter(old_x, old_y, s=0.2, c="tab:blue", alpha=0.7)  # plots the points as red dots

        return

    def get_final_result(self) -> pandas.core.frame.DataFrame:
        """get points assignment to each grid and transform the data into pandas df.

        Returns:
            results (DataFrame): A pandas dataframe containing the gridding information
        """

        # point_indexes_list = []
        point_grid_width_list = []
        point_grid_height_list = []
        point_grid_points_number_list = []
        calibration_point_list = []
        for grid in self.grids:
            # point_indexes_list.append([point.index for point in grid.points])
            point_grid_width_list.append(self.grid_len)
            point_grid_height_list.append(self.grid_len)
            point_grid_points_number_list.append(len(grid.points))
            calibration_point_list.append((round(grid.x_range[0], 6), round(grid.y_range[0], 6)))

        result = pd.DataFrame(
            {
                # "checklist_indexes": point_indexes_list,
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

        if self.plot_empty:
            pass
        else:
            result = result[result["stixel_checklist_count"] >= self.points_lower_threshold]
        return result
