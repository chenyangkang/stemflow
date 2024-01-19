# import math
# import os
# import warnings
# from collections.abc import Sequence

# # from multiprocessing import Pool
# from typing import Tuple, Union

# import matplotlib.patches as patches
# import matplotlib.pyplot as plt  # plotting libraries
# import numpy as np
# import pandas
# import pandas as pd

# from ..utils.generate_soft_colors import generate_soft_color
# from ..utils.sphere.coordinate_transform import lonlat_spherical_transformer
# from ..utils.sphere.distance import spherical_distance_from_coordinates
# from ..utils.sphere.Icosahedron import get_earth_Icosahedron_vertices_and_faces
# from ..utils.validation import check_random_state
# from .Q_blocks import Grid, Node, Point

# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"

# warnings.filterwarnings("ignore")


# def recursive_subdivide(
#     node: Node,
#     grid_len_upper_threshold: Union[float, int],
#     grid_len_lower_threshold: Union[float, int],
#     points_lower_threshold: Union[float, int],
# ):
#     """recursively subdivide the grids

#     Args:
#         node:
#             node input
#         grid_len_lon_upper_threshold:
#             force divide if grid longitude larger than the threshold
#         grid_len_lon_lower_threshold:
#             stop divide if grid longitude **will** be below than the threshold
#         grid_len_lat_upper_threshold:
#             force divide if grid latitude larger than the threshold
#         grid_len_lat_lower_threshold:
#             stop divide if grid latitude **will** be below than the threshold

#     """
#     raise NotImplementedError()

#     if node.width / 2 < grid_len_lower_threshold:
#         # The width and height will be the same. So only test one.
#         return

#     w_ = float(node.width / 2)

#     p = contains(node.x0, node.y0, w_, h_, node.points)
#     x1 = Node(node.x0, node.y0, w_, h_, p)
#     recursive_subdivide(
#         x1,
#         grid_len_lon_upper_threshold,
#         grid_len_lon_lower_threshold,
#         grid_len_lat_upper_threshold,
#         grid_len_lat_lower_threshold,
#         points_lower_threshold,
#     )

#     p = contains(node.x0, node.y0 + h_, w_, h_, node.points)
#     x2 = Node(node.x0, node.y0 + h_, w_, h_, p)
#     recursive_subdivide(
#         x2,
#         grid_len_lon_upper_threshold,
#         grid_len_lon_lower_threshold,
#         grid_len_lat_upper_threshold,
#         grid_len_lat_lower_threshold,
#         points_lower_threshold,
#     )

#     p = contains(node.x0 + w_, node.y0, w_, h_, node.points)
#     x3 = Node(node.x0 + w_, node.y0, w_, h_, p)
#     recursive_subdivide(
#         x3,
#         grid_len_lon_upper_threshold,
#         grid_len_lon_lower_threshold,
#         grid_len_lat_upper_threshold,
#         grid_len_lat_lower_threshold,
#         points_lower_threshold,
#     )

#     p = contains(node.x0 + w_, node.y0 + h_, w_, h_, node.points)
#     x4 = Node(node.x0 + w_, node.y0 + h_, w_, h_, p)
#     recursive_subdivide(
#         x4,
#         grid_len_lon_upper_threshold,
#         grid_len_lon_lower_threshold,
#         grid_len_lat_upper_threshold,
#         grid_len_lat_lower_threshold,
#         points_lower_threshold,
#     )

#     for ch_node in [x1, x2, x3, x4]:
#         if len(ch_node.points) <= points_lower_threshold:
#             if not ((node.width > grid_len_lon_upper_threshold) or (node.height > grid_len_lat_upper_threshold)):
#                 return

#     node.children = [x1, x2, x3, x4]


# def contains(x, y, w, h, points):
#     """return list of points within the grid"""
#     pts = []
#     for point in points:
#         if point.x >= x and point.x <= x + w and point.y >= y and point.y <= y + h:
#             pts.append(point)
#     return pts


# def find_children(node):
#     """return children nodes of this node"""
#     if not node.children:
#         return [node]
#     else:
#         children = []
#         for child in node.children:
#             children += find_children(child)
#     return children


# class Sphere_QTree:
#     """A spherical Quadtree class"""

#     def __init__(
#         self,
#         grid_len_upper_threshold: Union[float, int],
#         grid_len_lower_threshold: Union[float, int],
#         points_lower_threshold: int,
#         lon_lat_equal_grid: bool = True,
#         rotation_angle: Union[float, int] = 0,
#         calibration_point_x_jitter: Union[float, int] = 0,
#         calibration_point_y_jitter: Union[float, int] = 0,
#     ):
#         pass

#     def add_lon_lat_data(self, indexes: Sequence, x_array: Sequence, y_array: Sequence):
#         """Store input lng lat data and transform to **Point** object

#         Parameters:
#             indexes: Unique identifier for indexing the point.
#             x_array: longitudinal values.
#             y_array: latitudinal values.

#         """
#         if not len(x_array) == len(y_array) or not len(x_array) == len(indexes):
#             raise ValueError("input longitude and latitude and indexes not in same length!")

#         data = np.array([x_array, y_array]).T
#         angle = self.rotation_angle
#         r = angle / 360
#         theta = r * np.pi * 2
#         rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
#         data = data @ rotation_matrix
#         lon_new = (data[:, 0] + self.calibration_point_x_jitter).tolist()
#         lat_new = (data[:, 1] + self.calibration_point_y_jitter).tolist()

#         for index, lon, lat in zip(indexes, lon_new, lat_new):
#             self.points.append(Point(index, lon, lat))

#     def generate_gridding_params(self):
#         """For completeness"""
#         pass

#     def get_points(self):
#         """For completeness"""
#         pass
