from collections.abc import Sequence

import numpy as np

from ...gridding.Q_blocks import QPoint_3D


class lonlat_cartesian_3D_transformer:
    def __init__(self) -> None:
        pass

    def transform(lng, lat, radius=6371):
        # Convert latitude and longitude from degrees to radians
        lat_rad = np.radians(lat)
        lng_rad = np.radians(lng)

        # Calculate Cartesian coordinates
        x = radius * np.cos(lat_rad) * np.cos(lng_rad)
        y = radius * np.cos(lat_rad) * np.sin(lng_rad)
        z = radius * np.sin(lat_rad)

        return x, y, z

    def inverse_transform(x, y, z, r=None):
        if r is None:
            r = np.sqrt(x**2 + y**2 + z**2)
        latitude = np.degrees(np.arcsin(z / r))
        longitude = np.degrees(np.arctan2(y, x))
        return longitude, latitude


def get_midpoint_3D(p1, p2, radius=6371):
    v1 = np.array([p1.x, p1.y, p1.z])
    v2 = np.array([p2.x, p2.y, p2.z])

    v3 = v1 + v2
    v3 = v3 * (radius / np.linalg.norm(v3))

    p3 = QPoint_3D(None, v3[0], v3[1], v3[2])

    return p3


def continuous_interpolation_3D_plotting(p1, p2, radius=6371):
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
