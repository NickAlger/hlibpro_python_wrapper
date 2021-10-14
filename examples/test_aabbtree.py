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
b_mins = np.min(bb0, axis=2)
b_maxes = np.max(bb0, axis=2)

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


bad_qq = qq[bad_points, :]
T1 = np.any(np.all(b_mins[None, :, :] <= bad_qq[:, None, :], axis=2), axis=1)
T2 = np.any(np.all(bad_qq[:, None, :] <= b_maxes[None, :, :], axis=2), axis=1)
bad_points_that_are_in_a_box = np.logical_and(T1, T2)
all_bad_points_are_outside_all_boxes = (not np.any(bad_points_that_are_in_a_box))
print('all_bad_points_are_outside_all_boxes=', all_bad_points_are_outside_all_boxes)

bad_points_that_are_in_a_box

bad_qq_in_a_box = bad_qq[bad_points_that_are_in_a_box, :]
