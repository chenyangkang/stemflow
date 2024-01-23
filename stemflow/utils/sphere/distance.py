import numpy as np

from .coordinate_transform import lonlat_cartesian_3D_transformer


def distance_from_3D_point(x1, y1, z1, x2, y2, z2, radius=6371.0):
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


def haversine_distance(coord1, coord2, radius_earth):
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
    radius_earth = 6371.0

    # Calculate the distance
    distance = radius_earth * c

    return distance


# def haversine_distance(lon1, lat1, lon2, lat2, radius=6371.0):
#     """
#     Calculate the spherical distance between two points on the Earth's surface.

#     Args:
#         lat1 (float): Latitude of the first point in Radius.
#         lon1 (float): Longitude of the first point in Radius.
#         lat2 (float): Latitude of the second point in Radius.
#         lon2 (float): Longitude of the second point in Radius.
#         radius (float, optional): Radius of the Earth in kilometers (default is 6371.0).

#     Returns:
#         float: Spherical distance between the two points in kilometers.

#     """
#     # Calculate differences in coordinates
#     dlat = lat2 - lat1
#     dlon = lon2 - lon1

#     # Haversine formula
#     a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
#     c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

#     # Calculate distance
#     distance = radius * c

#     return distance
