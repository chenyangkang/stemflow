import numpy as np

# def sign(target, p2, p3):
#     return np.sign((target[:,0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (target[:,1] - p3[1]))

# def point_in_triangle(targets, p1, p2, p3):

#     d1 = sign(targets, p1, p2)
#     d2 = sign(targets, p2, p3)
#     d3 = sign(targets, p3, p1)

#     signs = np.column_stack([d1<0,d2<0,d3<0])
#     has_neg = signs.sum(axis=1)
#     has_pos = -signs.sum(axis=1)
#     return np.logical_not(np.logical_and(has_neg, has_pos))


def is_point_inside_triangle(point, A, B, C):
    u = np.cross(C - B, point - B) @ np.cross(C - B, A - B)
    v = np.cross(A - C, point - C) @ np.cross(A - C, B - C)
    w = np.cross(B - A, point - A) @ np.cross(B - A, C - A)

    return (u >= 0) & (v >= 0) & (w >= 0)


def intersect_triangle_plane(P0, V, A, B, C):
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
