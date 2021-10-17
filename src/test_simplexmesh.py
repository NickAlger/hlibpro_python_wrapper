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

coords = hcpp.projected_affine_coordinates_wrap_d2n3(query, points)

print('coords=', coords)

err_coords = np.linalg.norm(query - np.dot(points, coords))
print('err_coords=', err_coords)

err_affine_constraint = np.abs(1. - np.sum(coords))
print('err_affine_constraint=', err_affine_constraint)

-1.9099 * points[:,0] + 0.838161 * points[:,1] + 2.07174 * points[:,2] - query