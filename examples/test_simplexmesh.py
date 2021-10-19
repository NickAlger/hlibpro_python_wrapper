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


# CLOSEST POINT TO LINE SEGMENT

npts = 2
dim = 2
nquery = 100

segment_vertices = np.random.randn(dim, npts)

plt.figure()
plt.plot(segment_vertices[0,:], segment_vertices[1,:])

qq = np.random.randn(dim, nquery)
SS = segment_vertices.T.reshape((-1,1)) * np.ones(nquery)
pp = hcpp.closest_point_in_simplex_vectorized(qq, SS)

for k in range(30):
    query = qq[:,k]
    closest_point = pp[:,k]
    plt.plot([query[0], closest_point[0]], [query[1], closest_point[1]], 'gray')
    plt.plot(query[0], query[1], '*r')
    plt.plot(closest_point[0], closest_point[1], '.k')

plt.gca().set_aspect('equal')


# CLOSEST POINT TO TRIANGLE

npts = 3
dim = 2
nquery = 100

triangle_vertices = np.random.randn(dim, npts)

plt.figure()
plt.plot([triangle_vertices[0,0], triangle_vertices[0,1], triangle_vertices[0,2], triangle_vertices[0,0]],
         [triangle_vertices[1,0], triangle_vertices[1,1], triangle_vertices[1,2], triangle_vertices[1,0]])

qq = np.random.randn(dim, nquery)
SS = triangle_vertices.T.reshape((-1,1)) * np.ones(nquery)
pp = hcpp.closest_point_in_simplex_vectorized(qq, SS)

for k in range(nquery):
    query = qq[:,k]
    closest_point = pp[:,k]
    plt.plot([query[0], closest_point[0]], [query[1], closest_point[1]], 'gray')
    plt.plot(query[0], query[1], '*r')
    plt.plot(closest_point[0], closest_point[1], '.k')

plt.gca().set_aspect('equal')


# TIMING

npts = 3
dim = 2
nquery = int(1e6)

qq = np.random.randn(dim, nquery)
SS = np.random.randn(dim*npts, nquery)

t = time()
pp = hcpp.closest_point_in_simplex_vectorized(qq, SS)
dt_closest_point = time() - t
print('nquery=', nquery, ', dt_closest_point=', dt_closest_point)