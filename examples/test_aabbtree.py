import numpy as np
import hlibpro_python_wrapper as hpro
from time import time
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

from nalger_helper_functions import plot_ellipse, plot_rectangle

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

# More realistic setting, check correctness

num_boxes = int(1e3)
num_pts = int(1e4)

box_area = 1. / num_boxes
box_h = np.power(box_area, 1./K)

box_centers = np.random.randn(num_boxes, K)
box_widths = box_h * np.abs(np.random.randn(num_boxes, K))
b_mins = box_centers - box_widths
b_maxes = box_centers + box_widths

qq = np.random.randn(num_pts, K)

AABB = hcpp.AABBTree2D(b_mins, b_maxes)

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

# More realistic setting, timing

num_boxes = int(1e5)
num_pts = int(1e7)

print('num_boxes=', num_boxes, ', num_pts=', num_pts)

box_area = 1. / num_boxes
box_h = np.power(box_area, 1./K)

box_centers = np.random.randn(num_boxes, K)
box_widths = box_h * np.abs(np.random.randn(num_boxes, K))
b_mins = box_centers - box_widths
b_maxes = box_centers + box_widths

qq = np.random.randn(num_pts, K)

t = time()
AABB = hcpp.AABBTree2D(b_mins, b_maxes)
dt_build = time() - t
print('dt_build=', dt_build)

t = time()
box_inds = AABB.first_point_intersection_vectorized(qq)
dt_first_point_intersection_vectorized = time() - t
print('dt_first_point_intersection_vectorized=', dt_first_point_intersection_vectorized)

# Recursive:
# num_boxes= 100000 , num_pts= 10000000
# dt_build= 0.11991548538208008
# dt_first_point_intersection_vectorized= 5.488429069519043

# Iterative:
# num_boxes= 100000 , num_pts= 10000000
# dt_build= 0.13932013511657715
# dt_first_point_intersection_vectorized= 5.261875629425049


# Ball query

num_boxes = 351

box_area = 2. / num_boxes
box_h = np.power(box_area, 1./K)

box_centers = np.random.randn(num_boxes, K)
box_widths = box_h * np.abs(np.random.randn(num_boxes, K))
b_mins = box_centers - box_widths
b_maxes = box_centers + box_widths

AABB = hcpp.AABBTree2D(b_mins, b_maxes)

c = np.random.randn(K)
r = 8.0 * box_h * np.random.randn()

intersections = AABB.all_ball_intersections(c, r)

plt.figure()
for k in range(num_boxes):
    plot_rectangle(b_mins[k,:], b_maxes[k,:], facecolor='b')

for jj in intersections:
    plot_rectangle(b_mins[jj, :], b_maxes[jj, :], facecolor='r')

circle1 = plt.Circle(c, r, edgecolor='k', fill=False)
plt.gca().add_patch(circle1)

big_box_min = np.min(b_mins, axis=0)
big_box_max = np.max(b_maxes, axis=0)
plt.xlim(big_box_min[0], big_box_max[0])
plt.ylim(big_box_min[1], big_box_max[1])
plt.show()