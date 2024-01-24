from typing import Tuple, Union

import numpy as np

from .coordinate_transform import lonlat_cartesian_3D_transformer


def distance_from_3D_point(
    x1: float, y1: float, z1: float, x2: float, y2: float, z2: float, radius: float = 6371.0
) -> float:
    """Calculate the distance (km) between two 3D points

    Args:
        x1 (float): x1
        y1 (float): y1
        z1 (float): z1
        x2 (float): x2
        y2 (float): y2
        z2 (float): z2
        radius (float, optional): radius of earth. Defaults to 6371.0.

    Returns:
        float: distance in km.
    """

    # Convert Cartesian coordinates to spherical coordinates (latitude and longitude)
    lon1, lat1 = lonlat_cartesian_3D_transformer.inverse_transform(x1, y1, z1)
    lon2, lat2 = lonlat_cartesian_3D_transformer.inverse_transform(x2, y2, z2)

    # Haversine formula
    distance = haversine_distance((lon1, lat1), (lon2, lat2), radius_earth=radius)

    return distance


def spherical_distance_from_coordinates(inclination1, azimuth1, inclination2, azimuth2, radius=6371.0):
    """
    Calculate the spherical distance between two points on a sphere given their spherical coordinates.

    Args:
        radius (float): Radius of the sphere.
        inclination1 (float): Inclination angle of the first point in Radius.
        azimuth1 (float): Azimuth angle of the first point in Radius.
        inclination2 (float): Inclination angle of the second point in Radius.
        azimuth2 (float): Azimuth angle of the second point in Radius.

    Returns:
        float: Spherical distance between the two points in the same units as the sphere's radius.
    """

    # Calculate spherical distance using the law of cosines
    central_angle = np.arccos(
        np.sin(inclination1) * np.sin(inclination2)
        + np.cos(inclination1) * np.cos(inclination2) * np.cos(azimuth2 - azimuth1)
    )

    # Calculate distance using the sphere's radius
    distance = radius * central_angle

    return distance


def haversine_distance(coord1: Tuple[float, float], coord2: Tuple[float, float], radius_earth: float = 6371.0):
    """
    Calculate the Haversine distance between two sets of coordinates.

    Parameters:
        coord1 (tuple): (latitude, longitude) for the first point
        coord2 (tuple): (latitude, longitude) for the second point

    Returns:
        float: Haversine distance in kilometers
    """
    # Convert latitude and longitude from degrees to radians
    lon1, lat1 = np.radians(coord1)
    lon2, lat2 = np.radians(coord2)

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    # Radius of the Earth in kilometers (mean value)
    # Calculate the distance
    distance = radius_earth * c

    return distance
