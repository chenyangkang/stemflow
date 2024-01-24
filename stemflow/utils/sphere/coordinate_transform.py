from collections.abc import Sequence
from typing import Tuple, Union

import numpy as np

from ...gridding.Q_blocks import QPoint_3D


class lonlat_cartesian_3D_transformer:
    """Transformer between longitude,latitude and 3d dimension (x,y,z)."""

    def __init__(self) -> None:
        pass

    def transform(lng: np.ndarray, lat: np.ndarray, radius: float = 6371.0) -> Tuple[np.ndarray, np.ndarray]:
        """Transform lng, lat to x,y,z

        Args:
            lng (np.ndarray): lng
            lat (np.ndarray): lat
            radius (float, optional): radius of earth in km. Defaults to 6371.

        Returns:
            Tuple[np.ndarray, np.ndarray]: x,y,z
        """

        # Convert latitude and longitude from degrees to radians
        lat_rad = np.radians(lat)
        lng_rad = np.radians(lng)

        # Calculate Cartesian coordinates
        x = radius * np.cos(lat_rad) * np.cos(lng_rad)
        y = radius * np.cos(lat_rad) * np.sin(lng_rad)
        z = radius * np.sin(lat_rad)

        return x, y, z

    def inverse_transform(
        x: np.ndarray, y: np.ndarray, z: np.ndarray, r: float = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """transform x,y,z to lon, lat

        Args:
            x (np.ndarray): x
            y (np.ndarray): y
            z (np.ndarray): z
            r (float, optional): Radius of your spherical coordinate. If not given, calculate from x,y,z. Defaults to None.

        Returns:
            Tuple[np.ndarray, np.ndarray]: longitude, latitude
        """
        if r is None:
            r = np.sqrt(x**2 + y**2 + z**2)
        latitude = np.degrees(np.arcsin(z / r))
        longitude = np.degrees(np.arctan2(y, x))
        return longitude, latitude


def get_midpoint_3D(p1: QPoint_3D, p2: QPoint_3D, radius: float = 6371.0) -> QPoint_3D:
    """Get the mid-point of three QPoint_3D objet (vector)

    Args:
        p1 (QPoint_3D): p1
        p2 (QPoint_3D): p2
        radius (float, optional): radius of earth in km. Defaults to 6371.0.

    Returns:
        QPoint_3D: mid-point.
    """
    v1 = np.array([p1.x, p1.y, p1.z])
    v2 = np.array([p2.x, p2.y, p2.z])

    v3 = v1 + v2
    v3 = v3 * (radius / np.linalg.norm(v3))

    p3 = QPoint_3D(None, v3[0], v3[1], v3[2])

    return p3


def continuous_interpolation_3D_plotting(
    p1: np.ndarray, p2: np.ndarray, radius: float = 6371.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """interpolate 10 points on earth surface between the given two points. For plotting.

    Args:
        p1 (np.ndarray): p1
        p2 (np.ndarray): p2
        radius (float, optional): radius of earth in km. Defaults to 6371.0.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 10 x, 10 y, 10 z
    """
    v1 = np.array([p1[0], p1[1], p1[2]])
    v2 = np.array([p2[0], p2[1], p2[2]])

    x_ = []
    y_ = []
    z_ = []
    for bins_ in np.linspace(0, 1, 10):
        v3 = v1 * bins_ + v2 * (1 - bins_)
        v3 = v3 * (radius / np.linalg.norm(v3))

        x_.append(v3[0])
        y_.append(v3[1])
        z_.append(v3[2])

    return np.array(x_), np.array(y_), np.array(z_)
