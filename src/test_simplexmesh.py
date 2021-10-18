import numpy as np
import hlibpro_python_wrapper as hpro
from time import time
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

hcpp = hpro.hpro_cpp

npts = 3
dim = 2

points = np.random.randn(dim,npts)
query = np.random.randn(dim)
coords = np.zeros(npts)

hcpp.projected_affine_coordinates(query, points, coords)

print('coords=', coords)

err_coords = np.linalg.norm(query - np.dot(points, coords))
print('err_coords=', err_coords)

err_affine_constraint = np.abs(1. - np.sum(coords))
print('err_affine_constraint=', err_affine_constraint)


# CLOSEST POINT

npts = 2
dim = 2

segment_vertices = np.random.randn(dim, npts)

plt.figure()
plt.plot(segment_vertices[0,:], segment_vertices[1,:])

for k in range(20):
    query = np.random.randn(dim)
    closest_point = np.zeros(dim)
    hcpp.closest_point_in_simplex(query, segment_vertices, closest_point)


    plt.plot([query[0], closest_point[0]], [query[1], closest_point[1]], 'k')
    plt.plot(query[0], query[1], '*r')
    plt.plot(closest_point[0], closest_point[1], '.r')
    plt.gca().set_aspect('equal')
