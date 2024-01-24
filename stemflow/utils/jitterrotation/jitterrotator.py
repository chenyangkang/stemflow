from typing import Tuple, Union

import numpy as np


class JitterRotator:
    """2D jitter rotator."""

    def __init__():
        pass

    @classmethod
    def rotate_jitter(
        cls,
        x_array: np.ndarray,
        y_array: np.ndarray,
        rotation_angle: Union[int, float],
        calibration_point_x_jitter: Union[int, float],
        calibration_point_y_jitter: Union[int, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Rotate Normal lng, lat to jittered, rotated space

        Args:
            x_array (np.ndarray): input lng/x
            y_array (np.ndarray): input lat/y
            rotation_angle (Union[int, float]): rotation angle
            calibration_point_x_jitter (Union[int, float]): calibration_point_x_jitter
            calibration_point_y_jitter (Union[int, float]): calibration_point_y_jitter

        Returns:
            Tuple[np.ndarray, np.ndarray]: newx, newy
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
    ) -> Tuple[np.ndarray, np.ndarray]:
        """reverse jitter and rotation

        Args:
            x_array_rotated (np.ndarray): input lng/x
            y_array_rotated (np.ndarray): input lng/x
            rotation_angle (Union[int, float]): rotation angle
            calibration_point_x_jitter (Union[int, float]): calibration_point_x_jitter
            calibration_point_y_jitter (Union[int, float]): calibration_point_y_jitter

        Returns:
            Tuple[np.ndarray, np.ndarray]: newx, newy

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


class Sphere_Jitterrotator:
    """3D jitter rotator"""

    def __init__(self) -> None:
        pass

    def rotate_jitter(point: np.ndarray, axis: np.ndarray, angle: Union[float, int]) -> np.ndarray:
        """rotate_jitter 3d points

        Args:
            point (np.ndarray): shape of (X, 3)
            axis (np.ndarray): shape of (3,)
            angle (Union[float, int]): angle in degree

        Returns:
            np.ndarray: rotated_point
        """
        u = np.array(axis)
        u = u / np.linalg.norm(u)

        angle_ = angle * (np.pi / 180)
        cos_theta = np.cos(angle_)
        sin_theta = np.sin(angle_)

        rotation_matrix = np.array(
            [
                [
                    cos_theta + u[0] ** 2 * (1 - cos_theta),
                    u[0] * u[1] * (1 - cos_theta) - u[2] * sin_theta,
                    u[0] * u[2] * (1 - cos_theta) + u[1] * sin_theta,
                ],
                [
                    u[1] * u[0] * (1 - cos_theta) + u[2] * sin_theta,
                    cos_theta + u[1] ** 2 * (1 - cos_theta),
                    u[1] * u[2] * (1 - cos_theta) - u[0] * sin_theta,
                ],
                [
                    u[2] * u[0] * (1 - cos_theta) - u[1] * sin_theta,
                    u[2] * u[1] * (1 - cos_theta) + u[0] * sin_theta,
                    cos_theta + u[2] ** 2 * (1 - cos_theta),
                ],
            ]
        )

        rotated_point = np.dot(point, rotation_matrix)
        return rotated_point

    def inverse_rotate_jitter(point: np.ndarray, axis: np.ndarray, angle: Union[float, int]) -> np.ndarray:
        """inverse rotate_jitter 3d points

        Args:
            point (np.ndarray): shape of (X, 3)
            axis (np.ndarray): shape of (3,)
            angle (Union[float, int]): angle in degree

        Returns:
            np.ndarray: inverse rotated point
        """
        u = np.array(axis)
        u = u / np.linalg.norm(u)

        angle_ = -angle * (np.pi / 180)
        cos_theta = np.cos(angle_)
        sin_theta = np.sin(angle_)

        rotation_matrix = np.array(
            [
                [
                    cos_theta + u[0] ** 2 * (1 - cos_theta),
                    u[0] * u[1] * (1 - cos_theta) - u[2] * sin_theta,
                    u[0] * u[2] * (1 - cos_theta) + u[1] * sin_theta,
                ],
                [
                    u[1] * u[0] * (1 - cos_theta) + u[2] * sin_theta,
                    cos_theta + u[1] ** 2 * (1 - cos_theta),
                    u[1] * u[2] * (1 - cos_theta) - u[0] * sin_theta,
                ],
                [
                    u[2] * u[0] * (1 - cos_theta) - u[1] * sin_theta,
                    u[2] * u[1] * (1 - cos_theta) + u[0] * sin_theta,
                    cos_theta + u[2] ** 2 * (1 - cos_theta),
                ],
            ]
        )

        rotated_point = np.dot(point, rotation_matrix)
        return rotated_point
