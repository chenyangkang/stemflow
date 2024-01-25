import math
import os
import sys
import warnings
from collections.abc import Sequence

# from multiprocessing import Pool
from typing import Tuple, Union

import matplotlib.patches as patches
import matplotlib.pyplot as plt  # plotting libraries
import numpy as np
import pandas
import pandas as pd
import plotly.express as px
from tqdm import tqdm

from ..utils.generate_soft_colors import generate_soft_color
from ..utils.jitterrotation.jitterrotator import Sphere_Jitterrotator
from ..utils.sphere.coordinate_transform import (
    continuous_interpolation_3D_plotting,
    get_midpoint_3D,
    lonlat_cartesian_3D_transformer,
)
from ..utils.sphere.discriminant_formula import intersect_triangle_plane
from ..utils.sphere.distance import distance_from_3D_point
from ..utils.sphere.Icosahedron import get_earth_Icosahedron_vertices_and_faces_3D
from ..utils.validation import check_random_state
from .Q_blocks import QPoint_3D, Sphere_QTriangle

sys.setrecursionlimit(500000)

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

warnings.filterwarnings("ignore")


def Sphere_recursive_subdivide(
    node: Sphere_QTriangle,
    grid_len_upper_threshold: Union[float, int],
    grid_len_lower_threshold: Union[float, int],
    points_lower_threshold: Union[float, int],
    radius: Union[float, int] = 6371.0,
):
    """recursively subdivide the grids

    Args:
        node:
            node input
        grid_len_upper_threshold:
            force divide if grid larger than the threshold
        grid_len_lower_threshold:
            stop divide if grid **will** be below than the threshold
        points_lower_threshold:
            Stop splitting if fall short
        radius:
            radius of earth.

    """

    if node.length / 2 < grid_len_lower_threshold:
        # The width and height will be the same. So only test one.
        return

    if len(node.points) == 0:
        return

    if len(node.points) / 4 < grid_len_lower_threshold:
        return

    pm12 = get_midpoint_3D(node.p1, node.p2, radius)
    pm13 = get_midpoint_3D(node.p1, node.p3, radius)
    pm23 = get_midpoint_3D(node.p2, node.p3, radius)

    # 1.
    points_contained = Sphere_contains(node.points, pm12, pm23, node.p2)
    x1 = Sphere_QTriangle(
        pm12,
        pm23,
        node.p2,
        points_contained,
        distance_from_3D_point(pm12.x, pm12.y, pm12.z, pm23.x, pm23.y, pm23.z, radius),
        radius,
    )
    Sphere_recursive_subdivide(
        x1,
        grid_len_upper_threshold,
        grid_len_lower_threshold,
        points_lower_threshold,
    )

    # 2.
    points_contained = Sphere_contains(node.points, pm12, pm13, node.p1)
    x2 = Sphere_QTriangle(
        pm12,
        pm13,
        node.p1,
        points_contained,
        distance_from_3D_point(pm12.x, pm12.y, pm12.z, pm13.x, pm13.y, pm13.z, radius),
        radius,
    )
    Sphere_recursive_subdivide(
        x2,
        grid_len_upper_threshold,
        grid_len_lower_threshold,
        points_lower_threshold,
    )

    # 3.
    points_contained = Sphere_contains(node.points, pm23, pm13, node.p3)
    x3 = Sphere_QTriangle(
        pm23,
        pm13,
        node.p3,
        points_contained,
        distance_from_3D_point(pm13.x, pm13.y, pm13.z, pm23.x, pm23.y, pm23.z, radius),
        radius,
    )
    Sphere_recursive_subdivide(
        x3,
        grid_len_upper_threshold,
        grid_len_lower_threshold,
        points_lower_threshold,
    )

    # 3.
    points_contained = Sphere_contains(node.points, pm12, pm13, pm23)
    x4 = Sphere_QTriangle(
        pm12,
        pm13,
        pm23,
        points_contained,
        distance_from_3D_point(pm12.x, pm12.y, pm12.z, pm13.x, pm13.y, pm13.z, radius),
        radius,
    )
    Sphere_recursive_subdivide(
        x4,
        grid_len_upper_threshold,
        grid_len_lower_threshold,
        points_lower_threshold,
    )

    for ch_node in [x1, x2, x3, x4]:
        if len(ch_node.points) <= points_lower_threshold:
            if not (node.length > grid_len_upper_threshold):
                return

    node.children = [x1, x2, x3, x4]


def Sphere_contains(points, p1, p2, p3):
    """return list of points within the grid"""
    pts = []
    P0 = np.array([0, 0, 0]).reshape(1, -1)
    A = np.array([p1.x, p1.y, p1.z])
    B = np.array([p2.x, p2.y, p2.z])
    C = np.array([p3.x, p3.y, p3.z])

    V = np.array([[point.x, point.y, point.z] for point in points])

    intersect = intersect_triangle_plane(P0=P0, V=V, A=A, B=B, C=C)

    # print('intersect', intersect)
    pts = [points[i] for i in np.where(intersect)[0]]

    return pts


def Sphere_find_children(node):
    """return children nodes of this node"""
    if not node.children:
        return [node]
    else:
        children = []
        for child in node.children:
            children += Sphere_find_children(child)
    return children


class Sphere_QTree:
    """A spherical Quadtree class"""

    def __init__(
        self,
        grid_len_upper_threshold: Union[float, int],
        grid_len_lower_threshold: Union[float, int],
        points_lower_threshold: int,
        rotation_angle: Union[float, int] = None,
        rotation_axis: np.ndarray = None,
        radius: Union[float, int] = 6371,
        plot_empty: bool = False,
    ):
        """Create a Spherical QuadTree object

        Args:
            grid_len_upper_threshold:
                force divide if grid larger than the threshold
            grid_len_lower_threshold:
                stop divide if grid longitude **will** be below than the threshold
            points_lower_threshold:
                stop divide if points count is less than this threshold.
            rotation_angle:
                angles to rotate the gridding.
            rotation_axis:
                rotation_axis
            radius:
                radius of earth in km
            plot_empty:
                Whether to plot the empty grid

        Example:
            ```py
            >> QT_obj = Sphere_QTree(grid_len_upper_threshold=5000,
                            grid_len_lower_threshold=500,
                            points_lower_threshold=50,
                            rotation_angle = 15.5,
                            rotation_axis = np.array([-1,0,1]),
                            radius = 6371)
            >> QT_obj.add_lon_lat_data(sub_data.index, sub_data['longitude'].values, sub_data['latitude'].values)
            >> QT_obj.generate_gridding_params()
            >> QT_obj.subdivide() # Call subdivide to process
            >> gridding_info = QT_obj.get_final_result()  # gridding_info is a dataframe
            ```

        """

        self.points_lower_threshold = points_lower_threshold
        self.grid_len_upper_threshold = grid_len_upper_threshold
        self.grid_len_lower_threshold = grid_len_lower_threshold
        self.points = []
        if rotation_angle is None:
            rotation_angle = np.random.uniform(0, 90)
        self.rotation_angle = rotation_angle

        if rotation_axis is None:
            rotation_axis = np.random.uniform(-1, 1, 3)
        self.rotation_axis = rotation_axis
        self.radius = radius
        self.plot_empty = plot_empty

    def add_3D_data(self, indexes: Sequence, x_array: Sequence, y_array: Sequence, z_array: Sequence):
        """Store input x,y,z data and transform to **QPoint_3D** object

        Parameters:
            indexes: Unique identifier for indexing the point.
            x_array: x values.
            y_array: y values.
            z_array: z values
        """

        if not ((len(x_array) == len(indexes)) and (len(y_array) == len(indexes)) and (len(z_array) == len(indexes))):
            raise ValueError("input not in same length!")

        rotated_point = Sphere_Jitterrotator.rotate_jitter(
            np.column_stack([x_array, y_array, z_array]),
            self.rotation_axis,
            self.rotation_angle,
        )

        # print('rotated:' ,rotated_point)

        for index, x, y, z in zip(
            indexes, rotated_point[:, 0].flatten(), rotated_point[:, 1].flatten(), rotated_point[:, 2].flatten()
        ):
            self.points.append(QPoint_3D(index, x, y, z))

    def generate_gridding_params(self):
        """generate the gridding params after data are added"""
        self.root_list = []

        # 20 faces at the beginning
        vertices, faces = get_earth_Icosahedron_vertices_and_faces_3D(
            radius=6371
        )  # numpy array of shape (12,3,3) and (20,3,3)
        for face_index in range(faces.shape[0]):
            face = faces[face_index, :]

            face_obj = Sphere_QTriangle(
                p1=QPoint_3D(None, face[0, 0], face[0, 1], face[0, 2]),
                p2=QPoint_3D(None, face[1, 0], face[1, 1], face[1, 2]),
                p3=QPoint_3D(None, face[2, 0], face[2, 1], face[2, 2]),
                points=None,
                length=distance_from_3D_point(
                    face[0, 0], face[0, 1], face[0, 2], face[1, 0], face[1, 1], face[1, 2], self.radius
                ),
                radius=self.radius,
            )

            face_obj.points = Sphere_contains(self.points, face_obj.p1, face_obj.p2, face_obj.p3)
            self.root_list.append(face_obj)

    def get_points(self):
        """For completeness"""
        return self.points

    def subdivide(self, verbosity=0):
        """start recursively subdivide"""

        if verbosity > 0:
            for root_face in tqdm(self.root_list):
                Sphere_recursive_subdivide(
                    root_face,
                    self.grid_len_upper_threshold,
                    self.grid_len_lower_threshold,
                    self.points_lower_threshold,
                    self.radius,
                )
        else:
            for root_face in self.root_list:
                Sphere_recursive_subdivide(
                    root_face,
                    self.grid_len_upper_threshold,
                    self.grid_len_lower_threshold,
                    self.points_lower_threshold,
                    self.radius,
                )

    def get_final_result(self) -> pandas.core.frame.DataFrame:
        """get points assignment to each grid and transform the data into pandas df.

        Returns:
            results (DataFrame): A pandas dataframe containing the gridding information
        """
        all_grids = []
        for root_face in self.root_list:
            c = Sphere_find_children(root_face)
            all_grids += c

        # point_indexes_list = []
        point_grid_length_list = []
        point_grid_points_number_list = []
        p1x, p1y, p1z, p2x, p2y, p2z, p3x, p3y, p3z = [[] for i in range(9)]

        for grid in all_grids:
            # point_indexes_list.append([point.index for point in grid.points])
            point_grid_length_list.append(grid.length)
            point_grid_points_number_list.append(len(grid.points))
            p1x.append(round(grid.p1.x, 6))
            p1y.append(round(grid.p1.y, 6))
            p1z.append(round(grid.p1.z, 6))
            p2x.append(round(grid.p2.x, 6))
            p2y.append(round(grid.p2.y, 6))
            p2z.append(round(grid.p2.z, 6))
            p3x.append(round(grid.p3.x, 6))
            p3y.append(round(grid.p3.y, 6))
            p3z.append(round(grid.p3.z, 6))

        result = pd.DataFrame(
            {
                # "checklist_indexes": point_indexes_list,
                "stixel_indexes": list(range(len(point_grid_length_list))),
                "stixel_length": point_grid_length_list,
                "stixel_checklist_count": point_grid_points_number_list,
                "p1x": p1x,
                "p1y": p1y,
                "p1z": p1z,
                "p2x": p2x,
                "p2y": p2y,
                "p2z": p2z,
                "p3x": p3x,
                "p3y": p3y,
                "p3z": p3z,
                "rotation_angle": [self.rotation_angle] * len(point_grid_length_list),
                "rotaton_axis_x": [self.rotation_axis[0]] * len(point_grid_length_list),
                "rotaton_axis_y": [self.rotation_axis[1]] * len(point_grid_length_list),
                "rotaton_axis_z": [self.rotation_axis[2]] * len(point_grid_length_list),
            }
        )

        if self.plot_empty:
            pass
        else:
            result = result[result["stixel_checklist_count"] >= self.points_lower_threshold]
        return result

    def graph(self, scatter: bool = True, ax=None, line_kwgs={}):
        """plot gridding

        Args:
            scatter: Whether add scatterplot of data points
        """
        the_color = generate_soft_color()

        c = []
        for root_face in self.root_list:
            c += Sphere_find_children(root_face)

        for n in c:
            old_points = Sphere_Jitterrotator.inverse_rotate_jitter(
                np.array(
                    [
                        [n.p1.x, n.p1.y, n.p1.z],
                        [n.p2.x, n.p2.y, n.p2.z],
                        [n.p3.x, n.p3.y, n.p3.z],
                    ]
                ),
                self.rotation_axis,
                self.rotation_angle,
            )

            if ax is None:
                fig = plt.gcf()
                ax = fig.gca(projection="3d")

                ax.plot(
                    *continuous_interpolation_3D_plotting(old_points[0], old_points[1]), color=the_color, **line_kwgs
                )
                ax.plot(
                    *continuous_interpolation_3D_plotting(old_points[0], old_points[2]), color=the_color, **line_kwgs
                )
                ax.plot(
                    *continuous_interpolation_3D_plotting(old_points[1], old_points[2]), color=the_color, **line_kwgs
                )

            else:
                ax.plot(
                    *continuous_interpolation_3D_plotting(old_points[0], old_points[1]), color=the_color, **line_kwgs
                )
                ax.plot(
                    *continuous_interpolation_3D_plotting(old_points[0], old_points[2]), color=the_color, **line_kwgs
                )
                ax.plot(
                    *continuous_interpolation_3D_plotting(old_points[1], old_points[2]), color=the_color, **line_kwgs
                )

        if scatter:
            old_points = Sphere_Jitterrotator.inverse_rotate_jitter(
                np.column_stack(
                    [
                        [point.x for point in self.points],
                        [point.y for point in self.points],
                        [point.z for point in self.points],
                    ]
                ),
                self.rotation_axis,
                self.rotation_angle,
            )

            if ax is None:
                plt.scatter(
                    old_points[:, 0], old_points[:, 1], old_points[:, 2], s=0.2, c="tab:blue", alpha=0.7
                )  # plots the points as red dots
            else:
                ax.scatter(
                    old_points[:, 0], old_points[:, 1], old_points[:, 2], s=0.2, c="tab:blue", alpha=0.7
                )  # plots the points as red dots
        return

    def plotly_graph(self, scatter: bool = False, ax=None, line_kwgs={}):
        """Get plotly interactive plots

        Args:
            scatter (bool, optional): Whether to plot scatters. Defaults to False.
            ax (_type_, optional): Axes to plot on. Defaults to None.
            line_kwgs (dict, optional): line key words to pass to px.ling_geo. Defaults to {}.

        Returns:
            a plotly chart
        """
        the_color = generate_soft_color()
        this_slice = self.get_final_result()

        lats = []
        lons = []
        names = []

        from stemflow.utils.sphere.coordinate_transform import continuous_interpolation_3D_plotting

        for index, grid in this_slice.iterrows():
            # stixel_indexes = int(grid["stixel_indexes"])
            stixel_length = int(grid["stixel_length"])

            old_points = Sphere_Jitterrotator.inverse_rotate_jitter(
                np.array(
                    [
                        [grid["p1x"], grid["p1y"], grid["p1z"]],
                        [grid["p2x"], grid["p2y"], grid["p2z"]],
                        [grid["p3x"], grid["p3y"], grid["p3z"]],
                    ]
                ),
                self.rotation_axis,
                self.rotation_angle,
            )

            for ss in [[0, 1], [1, 2], [0, 2]]:
                the_lon, the_lat = lonlat_cartesian_3D_transformer.inverse_transform(
                    *continuous_interpolation_3D_plotting(old_points[ss[0]], old_points[ss[1]])
                )
                lons = np.append(lons, the_lon)
                lats = np.append(lats, the_lat)
                names = np.append(names, [f"{stixel_length}km"] * len(the_lon))
                lons = np.append(lons, None)
                lats = np.append(lats, None)
                names = np.append(names, None)

        lats_scatter = []
        lons_scatter = []
        names_scatter = []

        if scatter:
            old_points = Sphere_Jitterrotator.inverse_rotate_jitter(
                np.column_stack(
                    [
                        [point.x for point in self.points],
                        [point.y for point in self.points],
                        [point.z for point in self.points],
                    ]
                ),
                self.rotation_axis,
                self.rotation_angle,
            )

            the_lon, the_lat = lonlat_cartesian_3D_transformer.inverse_transform(
                old_points[:, 0], old_points[:, 1], old_points[:, 2]
            )
            lons_scatter = np.append(lons_scatter, the_lon)
            lats_scatter = np.append(lats_scatter, the_lat)
            names_scatter = np.append(names_scatter, [f"{stixel_length}km"] * len(the_lon))

        if ax is None:
            ax = px.line_geo(
                lat=lats,
                lon=lons,
                hover_name=names,
                projection="orthographic",
                width=1000,
                height=1000,
                color_discrete_sequence=[f"rgb({the_color[0]}, {the_color[1]}, {the_color[2]})"],
                **line_kwgs,
            )
            if scatter:
                ax.add_trace(
                    px.scatter_geo(
                        lat=lats_scatter, lon=lons_scatter, projection="orthographic", width=1000, height=1000
                    ).data[0]
                )
            return ax
        else:
            ax.add_trace(
                px.line_geo(
                    lat=lats,
                    lon=lons,
                    hover_name=names,
                    projection="orthographic",
                    width=1000,
                    height=1000,
                    color_discrete_sequence=[f"rgb({the_color[0]}, {the_color[1]}, {the_color[2]})"],
                    **line_kwgs,
                ).data[0]
            )
            if scatter:
                ax.add_trace(
                    px.scatter_geo(
                        lat=lats_scatter, lon=lons_scatter, projection="orthographic", width=1000, height=1000
                    ).data[0]
                )
            return ax
