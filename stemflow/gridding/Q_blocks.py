"""I call this Q_blocks because they are essential blocks for QTree methods"""

from collections.abc import Sequence
from typing import List, Tuple, Union

# from ..utils.sphere.coordinate_transform import lonlat_spherical_transformer
# from ..utils.sphere.distance import distance_from_3D_point


class QPoint:
    """A Point class for recording data points"""

    def __init__(self, index, x, y):
        self.x = x
        self.y = y
        self.index = index


class QNode:
    """A tree-like division node class"""

    def __init__(
        self,
        x0: Union[float, int],
        y0: Union[float, int],
        w: Union[float, int],
        h: Union[float, int],
        points: Sequence,
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


class QGrid:
    """Grid class for STEM (fixed gird size)"""

    def __init__(self, x_index, y_index, x_range, y_range):
        self.x_index = x_index
        self.y_index = y_index
        self.x_range = x_range
        self.y_range = y_range
        self.points = []


class QPoint_3D:
    """A 3D Point class for recording data points"""

    def __init__(self, index, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.index = index


class Sphere_QTriangle:
    """A tree-like division triangle node class for Sphere Quadtree"""

    def __init__(
        self, p1: QPoint_3D, p2: QPoint_3D, p3: QPoint_3D, points: Sequence, length: Union[float, int], radius=6371.0
    ):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.length = length

        self.points = points
        self.children = []

    def get_points(self):
        return self.points
