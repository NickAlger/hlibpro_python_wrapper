import numpy as np
import hlibpro_python_wrapper as hpro
from time import time
from scipy.spatial import KDTree

hcpp = hpro.hpro_cpp


K = 2

pp = np.random.randn(100,K)
KDT = hcpp.KDTree2D(list(pp))

#

print('one query, one neighbor:')
q = np.random.randn(K)

ind, dsq = KDT.nearest_neighbor(q)

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

print('many querys, one neighbor:')
num_querys = 63
qq = np.random.randn(num_querys, K)

inds, dsqq = KDT.nearest_neighbor_vectorized(np.array(qq.T, order='F'))

nearest_points = pp[inds,:]

nearest_inds = np.argmin(np.linalg.norm(pp[:,None,:] - qq[None,:,:], axis=2), axis=0)
nearest_points_true = pp[nearest_inds,:]
dsqq_true = np.linalg.norm(qq - nearest_points_true, axis=1)**2

err_nearest = np.linalg.norm(nearest_points - nearest_points_true)
print('err_nearest=', err_nearest)

err_dsqq = np.linalg.norm(dsqq - dsqq_true)
print('err_dsqq=', err_dsqq)

print('')

#

print('one querys, many neighbors:')
num_neighbors = 5
q = np.random.randn(K)

inds, dsqq = KDT.nearest_neighbor(q, num_neighbors)

nearest_points = pp[inds,:]

nearest_inds = np.argsort(np.linalg.norm(pp - q[None,:], axis=1), axis=0)[:num_neighbors]
nearest_points_true = pp[nearest_inds,:]
dsqq_true = np.linalg.norm(q[None,:] - nearest_points_true, axis=1)**2

err_nearest = np.linalg.norm(nearest_points - nearest_points_true)
print('err_nearest=', err_nearest)

err_dsqq = np.linalg.norm(dsqq - dsqq_true)
print('err_dsqq=', err_dsqq)

print('')

#

print('many querys, many neighbors:')
num_neighbors = 5
num_querys = 13
qq = np.random.randn(num_querys, K)

all_inds, all_dsqq = KDT.nearest_neighbor_vectorized(np.array(qq.T, order='F'), num_neighbors)

err_nearest = 0.0
err_dsqq = 0.0
for ii in range(num_querys):
    q = qq[ii, :]
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

pp = np.random.randn(n_pts, K)

t = time()
KDT = hcpp.KDTree2D(list(pp))
dt_build = time() - t
print('n_pts=', n_pts, ', dt_build=', dt_build)

qq = np.random.randn(n_query, K)
qq_T = np.array(qq.T, order='F')

t = time()
KDT.nearest_neighbor_vectorized(qq_T)
dt_query_one = time() - t
print('n_query=', n_query, ', dt_query_one=', dt_query_one)


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
KDT.nearest_neighbor_vectorized(qq_T, num_neighbors)
dt_query_many = time() - t
print('n_query=', n_query, ', num_neighbors=', num_neighbors, ', dt_query_many=', dt_query_many)

t = time()
KDT_scipy.query(qq, num_neighbors)
dt_query_many_scipy = time() - t
print('dt_query_many_scipy=', dt_query_many_scipy)

# timing:
# n_pts= 1000000 , dt_build= 2.382700204849243
# n_query= 1000000 , dt_query_one= 1.0918996334075928
# dt_build_scipy= 0.8152735233306885
# dt_query_one_scipy= 2.2504096031188965
# n_query= 1000000 , num_neighbors= 10 , dt_query_many= 3.2527642250061035
# dt_query_many_scipy= 4.948030233383179

####

# std::tuple<double, double>
# n_pts= 10000 , dt_build= 0.006009817123413086
# n_query= 10000000 , dt_query= 10.365345478057861
# dt_build_scipy= 0.0027518272399902344
# dt_query_scipy= 6.627758979797363

# Eigen Vector2d
# n_pts= 10000 , dt_build= 0.00654149055480957
# n_query= 10000000 , dt_query= 3.3113129138946533
# dt_build_scipy= 0.0027768611907958984
# dt_query_scipy= 6.6427671909332275

