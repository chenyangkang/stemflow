from typing import Union

import numpy as np


def is_point_inside_triangle(point: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Check if a point is inside a triangle

    Args:
        point (np.ndarray): point in vector. Shape (X, dimension).
        A (np.ndarray): point A of triangle. Shape (dimension).
        B (np.ndarray): point B of triangle. Shape (dimension).
        C (np.ndarray): point C of triangle. Shape (dimension).

    Returns:
        np.ndarray: inside or not
    """
    u = np.cross(C - B, point - B) @ np.cross(C - B, A - B)
    v = np.cross(A - C, point - C) @ np.cross(A - C, B - C)
    w = np.cross(B - A, point - A) @ np.cross(B - A, C - A)

    return (u >= 0) & (v >= 0) & (w >= 0)


def intersect_triangle_plane(P0: np.ndarray, V: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Get if the ray go through the triangle of A,B,C

    Args:
        P0 (np.ndarray): start point of ray
        V (np.ndarray): A point that the ray go through
        A (np.ndarray): point A of triangle. Shape (dimension).
        B (np.ndarray): point A of triangle. Shape (dimension).
        C (np.ndarray): point A of triangle. Shape (dimension).

    Returns:
        np.ndarray: Whether the point go through triangle ABC
    """
    # Calculate the normal vector of the plane
    N = np.cross(B - A, C - A)

    # A point on the plane
    P1 = A

    # Calculate the dot product of the normal vector and the ray direction
    # print(V.shape, N.shape)
    denom = np.dot(V, N)

    # Check if the vector is parallel to the plane
    para = abs(denom) < 1e-6

    # Calculate the parameter 't' to find the intersection point
    t = np.dot(P1 - P0, N) / denom

    # Check if the intersection point is along the ray
    intersect = (t >= 0) & (t <= 1)

    intersection_point = P0 + t.reshape(-1, 1) * V

    # Check if the intersection point is inside the triangle using the helper function
    inside = is_point_inside_triangle(intersection_point, A, B, C)

    return ~(para) & intersect & inside
