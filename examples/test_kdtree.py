import numpy as np
import hlibpro_python_wrapper as hpro
from time import time
from scipy.spatial import KDTree

hcpp = hpro.hpro_cpp


K = 2

pp = np.random.randn(100,K)
KDT = hcpp.KDTree2D(np.array(pp.T, order='F'))

#

print('one query, one neighbor:')
q = np.random.randn(K)

ind, dsq = KDT.query(q)

nearest_point = pp[ind,:]

nearest_ind = np.argmin(np.linalg.norm(pp - q[None,:], axis=1))
nearest_point_true = pp[nearest_ind,:]
dsq_true = np.linalg.norm(nearest_point_true - q)**2
err_nearest_one_point = np.linalg.norm(nearest_point - nearest_point_true)
err_dsq_one_point = np.abs(dsq - dsq_true)
print('err_nearest_one_point=', err_nearest_one_point)
print('err_dsq_one_point=', err_dsq_one_point)

print('')

#

print('many querys, many neighbors:')
num_neighbors = 5
num_querys = 13
qq = np.random.randn(num_querys, K)

all_inds, all_dsqq = KDT.query(np.array(qq.T, order='F'), num_neighbors)

err_nearest = 0.0
err_dsqq = 0.0
for ii in range(num_querys):
    q = qq[ii,:]
    inds = all_inds[:,ii]
    dsqq = all_dsqq[:,ii]
    nearest_points = pp[inds,:]

    nearest_inds = np.argsort(np.linalg.norm(pp - q[None,:], axis=1), axis=0)[:num_neighbors]
    nearest_points_true = pp[nearest_inds, :]
    dsqq_true = np.linalg.norm(q[None, :] - nearest_points_true, axis=1) ** 2

    err_nearest += np.linalg.norm(nearest_points - nearest_points_true)
    err_dsqq += np.linalg.norm(dsqq - dsqq_true)

print('err_nearest=', err_nearest)
print('err_dsqq=', err_dsqq)

print('')

#

print('timing:')
n_pts = int(1e6)
n_query = int(1e6)
num_neighbors = 10
print('n_pts=', n_pts, ', n_query=', n_query, ', num_neighbors=', num_neighbors)

pp = np.random.randn(n_pts, K)
pp_T = np.array(pp.T, order='F')

t = time()
KDT = hcpp.KDTree2D(pp_T)
dt_build = time() - t
print('dt_build=', dt_build)

qq = np.random.randn(n_query, K)
qq_T = np.array(qq.T, order='F')

t = time()
KDT.query(qq_T, 1)
dt_query_one = time() - t
print('dt_query_one=', dt_query_one)


t = time()
KDT_scipy = KDTree(pp)
dt_build_scipy = time() - t
print('dt_build_scipy=', dt_build_scipy)

t = time()
KDT_scipy.query(qq)
dt_query_one_scipy = time() - t
print('dt_query_one_scipy=', dt_query_one_scipy)

#

t = time()
KDT.query(qq_T, num_neighbors)
dt_query_many = time() - t
print('dt_query_many=', dt_query_many)

t = time()
KDT_scipy.query(qq, num_neighbors)
dt_query_many_scipy = time() - t
print('dt_query_many_scipy=', dt_query_many_scipy)


# Resulting output 11/9/21:
#
# one query, one neighbor:
# err_nearest_one_point= 0.0
# err_dsq_one_point= 2.7755575615628914e-17
#
# many querys, many neighbors:
# err_nearest= 0.0
# err_dsqq= 8.930789300493098e-16
#
# timing:
# n_pts= 1000000 , n_query= 1000000 , num_neighbors= 10
# dt_build= 1.491417646408081
# dt_query_one= 1.1302480697631836
# dt_build_scipy= 0.7670538425445557
# dt_query_one_scipy= 2.1079294681549072
# dt_query_many= 3.0853629112243652
# dt_query_many_scipy= 4.660918712615967
