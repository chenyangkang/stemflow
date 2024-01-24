"Functions for the initial icosahedron in spherical indexing system"

import numpy as np

from .coordinate_transform import lonlat_cartesian_3D_transformer


def get_Icosahedron_vertices() -> np.ndarray:
    """Return the 12 vertices of icosahedron

    Returns:
        np.ndarray: (n_vertices, 3D_coordinates)
    """
    phi = (1 + np.sqrt(5)) / 2
    vertices = np.array(
        [
            (phi, 1, 0),
            (phi, -1, 0),
            (-phi, -1, 0),
            (-phi, 1, 0),
            (1, 0, phi),
            (-1, 0, phi),
            (-1, 0, -phi),
            (1, 0, -phi),
            (0, phi, 1),
            (0, phi, -1),
            (0, -phi, -1),
            (0, -phi, 1),
        ]
    )
    return vertices


def calc_and_judge_distance(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> bool:
    """Determine if the three points have same distance with each other

    Args:
        v1 (np.ndarray): point 1
        v2 (np.ndarray): point 1
        v3 (np.ndarray): point 1

    Returns:
        bool: Whether have same pair-wise distance
    """
    d1 = np.sum((np.array(v1) - np.array(v2)) ** 2) ** (1 / 2)
    d2 = np.sum((np.array(v1) - np.array(v3)) ** 2) ** (1 / 2)
    d3 = np.sum((np.array(v2) - np.array(v3)) ** 2) ** (1 / 2)
    if d1 == d2 == d3 == 2:
        return True
    else:
        return False


def get_Icosahedron_faces() -> np.ndarray:
    """Get icosahedron faces

    Returns:
        np.ndarray: shape (20,3,3). (faces, point, 3d_dimension)
    """
    vertices = get_Icosahedron_vertices()

    face_list = []
    for vertice1 in vertices:
        for vertice2 in vertices:
            for vertice3 in vertices:
                same_face = calc_and_judge_distance(vertice1, vertice2, vertice3)
                if same_face:
                    the_face_set = set([tuple(vertice1), tuple(vertice2), tuple(vertice3)])
                    if the_face_set not in face_list:
                        face_list.append(the_face_set)

    face_list = np.array([list(i) for i in face_list])
    return face_list


def get_earth_Icosahedron_vertices_and_faces_lonlat() -> [np.ndarray, np.ndarray]:
    """Get vertices and faces in lon, lat

    Returns:
        [np.ndarray, np.ndarray]: vertices, faces
    """
    # earth_radius_km=6371.0
    # get Icosahedron vertices and faces
    vertices = get_Icosahedron_vertices()
    face_list = get_Icosahedron_faces()

    # Scale: from 2 to 6371
    vertices_lng, vertices_lat = lonlat_cartesian_3D_transformer.inverse_transform(
        vertices[:, 0], vertices[:, 1], vertices[:, 2]
    )
    faces_lng, faces_lat = lonlat_cartesian_3D_transformer.inverse_transform(
        face_list[:, :, 0], face_list[:, :, 1], face_list[:, :, 2]
    )

    return np.stack([vertices_lng, vertices_lat], axis=-1), np.stack([faces_lng, faces_lat], axis=-1)


def get_earth_Icosahedron_vertices_and_faces_3D(radius=1) -> [np.ndarray, np.ndarray]:
    """Get vertices and faces in lon, lat

    Args:
        radius (Union[int, float]): radius of earth in km.

    Returns:
        [np.ndarray, np.ndarray]: vertices, faces
    """

    # earth_radius_km=6371.0
    # get Icosahedron vertices and faces
    vertices = get_Icosahedron_vertices()
    face_list = get_Icosahedron_faces()

    # Scale: from 2 to 6371
    scale_ori = (np.sum(vertices**2, axis=1) ** (1 / 2))[0]
    # Scale vertices and face_list to km
    vertices = vertices * (radius / scale_ori)
    face_list = face_list * (radius / scale_ori)

    return vertices, face_list
