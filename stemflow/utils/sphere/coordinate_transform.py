from collections.abc import Sequence

import numpy as np


class lonlat_spherical_transformer:
    def __init__(self) -> None:
        pass

    def fit(self):
        pass

    @staticmethod
    def lat_lon_to_spherical(longitude, latitude, radius=1.0):
        """
        Convert latitude and longitude to spherical coordinates.

        Args:
          longitude (float or numpy.ndarray): Longitude in degrees.
          latitude (float or numpy.ndarray): Latitude in degrees.
          radius (float, optional): Radius of the sphere (default is 1.0).

        Returns:
          tuple: Spherical coordinates (radius, inclination, azimuth).
        """
        # Convert degrees to radians
        lat_rad = np.radians(latitude)
        lon_rad = np.radians(longitude)

        # Calculate spherical coordinates
        x = radius * np.cos(lat_rad) * np.cos(lon_rad)
        y = radius * np.cos(lat_rad) * np.sin(lon_rad)
        z = radius * np.sin(lat_rad)

        # Calculate inclination (θ) and azimuth (φ)
        inclination = np.arccos(z / radius)
        azimuth = np.arctan2(y, x)

        return radius, inclination, azimuth

    @staticmethod
    def spherical_to_lat_lon(radius, inclination, azimuth):
        """
        Convert spherical coordinates to latitude and longitude.

        Args:
            radius (float): Radius of the sphere.
            inclination (float): Inclination angle in degrees.
            azimuth (float): Azimuth angle in degrees.

        Returns:
            tuple: Latitude and longitude in degrees.

        Examples:
            # Example usage:
            spherical_coordinates = (1.0, 45.0, 60.0)  # Example spherical coordinates
            lat_lon = spherical_to_lat_lon(*spherical_coordinates)
            print(lat_lon)
        """

        # Calculate Cartesian coordinates
        x = radius * np.sin(inclination) * np.cos(azimuth)
        y = radius * np.sin(inclination) * np.sin(azimuth)
        z = radius * np.cos(inclination)

        # Convert Cartesian coordinates to latitude and longitude
        longitude = np.degrees(np.arctan2(y, x))
        latitude = np.degrees(np.arcsin(z / radius))

        return longitude, latitude

    @classmethod
    def transform(self, lng: Sequence, lat: Sequence) -> [Sequence, Sequence]:
        """From lng, lat to inclination, azimuth (in radian)"""

        radius, inclination, azimuth = self.lat_lon_to_spherical(lng, lat)
        return inclination, azimuth

    @classmethod
    def inverse_transform(self, inclination: Sequence, azimuth: Sequence, radius=1) -> [Sequence, Sequence]:
        """From inclination, azimuth (in radian) to lng, lat"""

        return self.spherical_to_lat_lon(radius, inclination, azimuth)


def cartesian_3D_to_lonlat(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    latitude = np.degrees(np.arcsin(z / r))
    longitude = np.degrees(np.arctan2(y, x))

    return longitude, latitude
