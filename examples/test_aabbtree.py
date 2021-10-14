import numpy as np
import hlibpro_python_wrapper as hpro
from time import time
from scipy.spatial import KDTree

hcpp = hpro.hpro_cpp

K=2
num_boxes = 10000
num_pts = 4987

# Random boxes might be inside out. Doing this to check code can handle this without segfaulting
b_mins0 = np.random.randn(num_boxes, K)
b_maxes0 = np.random.randn(num_boxes, K)

AABB = hcpp.AABBTree2D(b_mins0, b_maxes0)

q = np.random.randn(K)
ind = AABB.first_point_intersection(q)

qq = np.random.randn(num_boxes, K)
inds = AABB.first_point_intersection_vectorized(qq)

# Real boxes

num_boxes = 191
num_pts = 549

bb0 = np.random.randn(num_boxes, K, 2)
b_mins = np.min(bb0, axis=2).copy()
b_maxes = np.max(bb0, axis=2).copy()

AABB = hcpp.AABBTree2D(b_mins, b_maxes)
qq = 2.5 * np.random.randn(num_pts, K)
box_inds = AABB.first_point_intersection_vectorized(qq)

good_points = (box_inds >= 0)
bad_points = np.logical_not(good_points)

good_qq = qq[good_points,:]
S1 = b_mins[box_inds[good_points], :] <= good_qq
S2 = good_qq <= b_maxes[box_inds[good_points], :]
good_points_that_are_in_their_box = np.logical_and(S1, S2)
all_good_points_are_in_their_boxes = np.all(good_points_that_are_in_their_box)
print('all_good_points_are_in_their_boxes=', all_good_points_are_in_their_boxes)


points_that_are_in_at_least_one_box = np.any(np.all(np.logical_and(b_mins[None, :, :] <= qq[:, None, :],
                                                                   qq[:, None, :] <= b_maxes[None, :, :]), axis=2), axis=1)

points_that_are_not_in_any_box = np.logical_not(points_that_are_in_at_least_one_box)
bad_points_are_the_points_outside_all_boxes = np.all(points_that_are_not_in_any_box == bad_points)
print('bad_points_are_the_points_outside_all_boxes=', bad_points_are_the_points_outside_all_boxes)
