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
from .Q_blocks import QNode, QPoint

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

warnings.filterwarnings("ignore")


def recursive_subdivide(
    node: QNode,
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
    x1 = QNode(node.x0, node.y0, w_, h_, p)
    recursive_subdivide(
        x1,
        grid_len_lon_upper_threshold,
        grid_len_lon_lower_threshold,
        grid_len_lat_upper_threshold,
        grid_len_lat_lower_threshold,
        points_lower_threshold,
    )

    p = contains(node.x0, node.y0 + h_, w_, h_, node.points)
    x2 = QNode(node.x0, node.y0 + h_, w_, h_, p)
    recursive_subdivide(
        x2,
        grid_len_lon_upper_threshold,
        grid_len_lon_lower_threshold,
        grid_len_lat_upper_threshold,
        grid_len_lat_lower_threshold,
        points_lower_threshold,
    )

    p = contains(node.x0 + w_, node.y0, w_, h_, node.points)
    x3 = QNode(node.x0 + w_, node.y0, w_, h_, p)
    recursive_subdivide(
        x3,
        grid_len_lon_upper_threshold,
        grid_len_lon_lower_threshold,
        grid_len_lat_upper_threshold,
        grid_len_lat_lower_threshold,
        points_lower_threshold,
    )

    p = contains(node.x0 + w_, node.y0 + h_, w_, h_, node.points)
    x4 = QNode(node.x0 + w_, node.y0 + h_, w_, h_, p)
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
        plot_empty: bool = False,
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
            plot_empty:
                Whether to plot the empty grid

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

        lon_new, lat_new = JitterRotator.rotate_jitter(
            x_array, y_array, self.rotation_angle, self.calibration_point_x_jitter, self.calibration_point_y_jitter
        )

        for index, lon, lat in zip(indexes, lon_new, lat_new):
            self.points.append(QPoint(index, lon, lat))

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
            self.root = QNode(
                left_bottom_point_x,
                left_bottom_point_y,
                max(self.grid_length_x, self.grid_length_y),
                max(self.grid_length_x, self.grid_length_y),
                self.points,
            )
        elif self.lon_lat_equal_grid is False:
            self.root = QNode(
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

        for n in c:
            old_x, old_y = JitterRotator.inverse_jitter_rotate(
                [n.x0], [n.y0], self.rotation_angle, self.calibration_point_x_jitter, self.calibration_point_y_jitter
            )

            if ax is None:
                plt.gcf().gca().add_patch(
                    patches.Rectangle(
                        (old_x, old_y), n.width, n.height, fill=False, angle=self.rotation_angle, color=the_color
                    )
                )
            else:
                ax.add_patch(
                    patches.Rectangle(
                        (old_x, old_y), n.width, n.height, fill=False, angle=self.rotation_angle, color=the_color
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
        all_grids = find_children(self.root)
        # point_indexes_list = []
        point_grid_width_list = []
        point_grid_height_list = []
        point_grid_points_number_list = []
        calibration_point_list = []
        for grid in all_grids:
            # point_indexes_list.append([point.index for point in grid.points])
            point_grid_width_list.append(grid.width)
            point_grid_height_list.append(grid.height)
            point_grid_points_number_list.append(len(grid.points))
            calibration_point_list.append((round(grid.x0, 6), round(grid.y0, 6)))

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
