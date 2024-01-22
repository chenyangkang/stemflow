from typing import Tuple, Union

import numpy as np

# import geopandas as gpd


class JitterRotator:
    def __init__():
        pass

    # @classmethod
    # def rotate_jitter_gpd(cls,
    #                       df: gpd.geodataframe.GeoDataFrame,
    #                       rotation_angle: Union[int, float],
    #                       calibration_point_x_jitter: Union[int, float],
    #                       calibration_point_y_jitter: Union[int, float]
    #                       ) -> gpd.geodataframe.GeoDataFrame:
    #     """Rotate Normal lng, lat to jittered, rotated space

    #     Args:
    #         x_array (np.ndarray): input lng/x
    #         y_array (np.ndarray): input lat/y
    #         rotation_angle (Union[int, float]): rotation angle
    #         calibration_point_x_jitter (Union[int, float]): calibration_point_x_jitter
    #         calibration_point_y_jitter (Union[int, float]): calibration_point_y_jitter

    #     Returns:
    #         tuple(np.ndarray, np.ndarray): newx, newy
    #     """
    #     transformed_series = df.rotate(
    #         rotation_angle, origin=(0,0)
    #     ).affine_transform(
    #         [1,0,0,1,calibration_point_x_jitter,calibration_point_y_jitter]
    #     )

    #     df1 = gpd.GeoDataFrame(df, geometry=transformed_series)

    #     return df1

    # @classmethod
    # def inverse_jitter_rotate_gpd(cls,
    #                       df_rotated: gpd.geodataframe.GeoDataFrame,
    #                       rotation_angle: Union[int, float],
    #                       calibration_point_x_jitter: Union[int, float],
    #                       calibration_point_y_jitter: Union[int, float]
    #                       ) -> gpd.geodataframe.GeoDataFrame:
    #     """reverse jitter and rotation

    #     Args:
    #         x_array_rotated (np.ndarray): input lng/x
    #         y_array_rotated (np.ndarray): input lng/x
    #         rotation_angle (Union[int, float]): rotation angle
    #         calibration_point_x_jitter (Union[int, float]): calibration_point_x_jitter
    #         calibration_point_y_jitter (Union[int, float]): calibration_point_y_jitter
    #     """

    #     return df_rotated.affine_transform(
    #         [1,0,0,1,-calibration_point_x_jitter,-calibration_point_y_jitter]
    #     ).rotate(
    #         -rotation_angle, origin=(0,0)
    #     )

    @classmethod
    def rotate_jitter(
        cls,
        x_array: np.ndarray,
        y_array: np.ndarray,
        rotation_angle: Union[int, float],
        calibration_point_x_jitter: Union[int, float],
        calibration_point_y_jitter: Union[int, float],
    ):
        """Rotate Normal lng, lat to jittered, rotated space

        Args:
            x_array (np.ndarray): input lng/x
            y_array (np.ndarray): input lat/y
            rotation_angle (Union[int, float]): rotation angle
            calibration_point_x_jitter (Union[int, float]): calibration_point_x_jitter
            calibration_point_y_jitter (Union[int, float]): calibration_point_y_jitter

        Returns:
            tuple(np.ndarray, np.ndarray): newx, newy
        """
        data = np.array([x_array, y_array]).T
        angle = rotation_angle
        r = angle / 360
        theta = r * np.pi * 2
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        data = data @ rotation_matrix
        lon_new = (data[:, 0] + calibration_point_x_jitter).tolist()
        lat_new = (data[:, 1] + calibration_point_y_jitter).tolist()
        return lon_new, lat_new

    @classmethod
    def inverse_jitter_rotate(
        cls,
        x_array_rotated: np.ndarray,
        y_array_rotated: np.ndarray,
        rotation_angle: Union[int, float],
        calibration_point_x_jitter: Union[int, float],
        calibration_point_y_jitter: Union[int, float],
    ):
        """reverse jitter and rotation

        Args:
            x_array_rotated (np.ndarray): input lng/x
            y_array_rotated (np.ndarray): input lng/x
            rotation_angle (Union[int, float]): rotation angle
            calibration_point_x_jitter (Union[int, float]): calibration_point_x_jitter
            calibration_point_y_jitter (Union[int, float]): calibration_point_y_jitter
        """
        theta = -(rotation_angle / 360) * np.pi * 2
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        back_jitter_data = np.array(
            [
                np.array(x_array_rotated) - calibration_point_x_jitter,
                np.array(y_array_rotated) - calibration_point_y_jitter,
            ]
        ).T
        back_rotated = back_jitter_data @ rotation_matrix
        return back_rotated[:, 0].flatten(), back_rotated[:, 1].flatten()
