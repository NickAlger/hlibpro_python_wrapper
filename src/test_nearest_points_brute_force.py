import numpy as np
import matplotlib.pyplot as plt
from time import time
import hlibpro_python_wrapper as hpro

hcpp = hpro.hpro_cpp

dim = 2
num_pts = 5
num_queries = 7
pp = np.array(np.random.randn(dim, num_pts), order='F')
qq = np.array(np.random.randn(dim, num_queries), order='F')
k = 3

nearest_inds = hcpp.nearest_points_brute_force_vectorized(pp, qq, k)

nearest_inds_true = np.zeros((k, num_queries), dtype=int)
for ii in range(num_queries):
    pairwise_distances = np.linalg.norm(pp - qq[:,ii].reshape((-1,1)), axis=0)
    sort_inds = np.argsort(pairwise_distances).reshape(-1)
    nearest_inds_true[:,ii] = sort_inds[:k]

err = np.linalg.norm(nearest_inds - nearest_inds_true)
print('err=', err)

# timing

dim = 2
num_pts = int(1e4)
num_queries = int(1e3)
pp = np.array(np.random.randn(dim, num_pts), order='F')
qq = np.array(np.random.randn(dim, num_queries), order='F')
k = 10

t = time()
nearest_inds = hcpp.nearest_points_brute_force_vectorized(pp, qq, k)
dt = time() - t
print('dim=', dim, ', num_pts', num_pts, ', num_queries=', num_queries, ', dt=', dt)