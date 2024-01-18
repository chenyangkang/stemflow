from typing import Tuple, Union


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


class Grid:
    """Grid class for STEM (fixed gird size)"""

    def __init__(self, x_index, y_index, x_range, y_range):
        self.x_index = x_index
        self.y_index = y_index
        self.x_range = x_range
        self.y_range = y_range
        self.points = []
