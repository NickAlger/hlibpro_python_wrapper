import numpy as np


class Node:
    def __init__(me, location, left_child, right_child):
        me.location = location
        me.left_child = left_child
        me.right_child = right_child


def kdtree(pp, depth):
    if pp.shape[0] < 1:
        return None

    num_pts, dim = pp.shape
    axis = depth % dim

    pp = pp[np.argsort(pp[:,axis]),:] # might be a lot of work in c++
    mid_ind = num_pts // 2

    left_pts = pp[: mid_ind, :]
    mid_pt = pp[mid_ind, :]
    right_pts = pp[mid_ind+1 :]

    left_kdtree = kdtree(left_pts, depth+1)
    right_kdtree = kdtree(right_pts, depth + 1)

    return Node(mid_pt, left_kdtree, right_kdtree)


def nearest_neighbor(query_point, tree_node, depth):
    best_point = tree_node.location
    best_distance = np.linalg.norm(query_point - tree_node.location)

    dim = query_point.size
    axis = depth % dim
    displacement_to_splitting_plane = tree_node.location[axis] - query_point[axis]
    if displacement_to_splitting_plane >= 0:
        child_A = tree_node.left_child
        child_B = closest_child = tree_node.right_child
    else:
        child_A = tree_node.right_child
        child_B = closest_child = tree_node.left_child

    if child_A is not None:
        point_A, distance_A = nearest_neighbor(query_point, child_A, depth+1)
        if distance_A < best_distance:
            best_point = point_A
            best_distance = distance_A

    if child_B is not None:
        B_distance_lower_bound = np.abs(displacement_to_splitting_plane)

        if B_distance_lower_bound < best_distance:
            point_B, distance_B = nearest_neighbor(query_point, child_B, depth+1)
            if distance_B < best_distance:
                best_point = point_B
                best_distance = distance_B

    return best_point, best_distance







pp = np.random.randn(100, 2)
root = kdtree(pp, 0)

q = np.random.randn(2)
print('q=', q)

p_nearest, dist_nearest = nearest_neighbor(q, root, 0)
print('p_nearest=', p_nearest)
print('dist_nearest=', dist_nearest)


nearest_ind = np.argmin(np.linalg.norm(pp - q, axis=1))
p_nearest_true = pp[nearest_ind, :]
dist_nearest_true = np.linalg.norm(p_nearest_true - q)
print('p_nearest_true=', p_nearest_true)
print('dist_nearest_true=', dist_nearest_true)