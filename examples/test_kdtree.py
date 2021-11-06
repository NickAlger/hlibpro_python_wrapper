import numpy as np
import hlibpro_python_wrapper as hpro
from time import time
from scipy.spatial import KDTree

hcpp = hpro.hpro_cpp


K = 2

def make_KDT(pp):
    dim = pp.shape[0]
    if dim == 1:
        KDT = hcpp.KDTree1D(pp)
    elif dim == 2:
        KDT = hcpp.KDTree2D(pp)
    elif dim == 3:
        KDT = hcpp.KDTree3D(pp)
    elif dim == 4:
        KDT = hcpp.KDTree4D(pp)
    else:
        raise RuntimeError('KDT only implemented for K<=4')
    return KDT

pp = np.array(np.random.randn(K, 100), order='F')
KDT = make_KDT(pp)

q = np.random.randn(K)

ind, dsq = KDT.nearest_neighbor(q)
nearest_point = pp[:,ind]

nearest_ind = np.argmin(np.linalg.norm(pp - q[:,None], axis=0))
nearest_point_true = pp[:,nearest_ind]
dsq_true = np.linalg.norm(nearest_point_true - q)**2
err_nearest_one_point = np.linalg.norm(nearest_point - nearest_point_true)
err_dsq_one_point = np.abs(dsq - dsq_true)
print('err_nearest_one_point=', err_nearest_one_point)
print('err_dsq_one_point=', err_dsq_one_point)

qq = np.array(np.random.randn(K, 77), order='F')
inds, dsqq = KDT.nearest_neighbor_vectorized(qq)
nearest_points = pp[:, inds]

nearest_inds = np.argmin(np.linalg.norm(pp[:,:,None] - qq[:,None,:], axis=0), axis=0)
nearest_points_true = pp[:,nearest_inds]
dsqq_true = np.linalg.norm(qq - nearest_points_true, axis=0)**2

err_nearest = np.linalg.norm(nearest_points - nearest_points_true)
print('err_nearest=', err_nearest)

err_dsqq = np.linalg.norm(dsqq - dsqq_true)
print('err_dsqq=', err_dsqq)



n_pts = int(1e6)
n_query = int(1e6)

pp = np.array(np.random.randn(K, n_pts), order='F')
t = time()
KDT = make_KDT(pp)
dt_build = time() - t
print('n_pts=', n_pts, ', dt_build=', dt_build)

qq = np.array(np.random.randn(K, n_query), order='F')
t = time()
KDT.nearest_neighbor_vectorized(qq)
dt_query = time() - t
print('n_query=', n_query, ', dt_query=', dt_query)


t = time()
KDT_scipy = KDTree(pp.T)
dt_build_scipy = time() - t
print('dt_build_scipy=', dt_build_scipy)

t = time()
KDT_scipy.query(qq.T)
dt_query_scipy = time() - t
print('dt_query_scipy=', dt_query_scipy)



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


# test multiple nearest neighbors

num_neighbors = 6

pp = np.array(np.random.randn(K,100), order='F')
KDT = make_KDT(pp)

q = np.random.randn(K)

inds, dsqs = KDT.nearest_neighbors(q, num_neighbors)
nearest_points = pp[:,inds]

nearest_inds = np.argsort(np.linalg.norm(pp - q[:,None], axis=0))
nearest_points_true = pp[:,nearest_inds[:num_neighbors]]
dsqs_true = np.linalg.norm(nearest_points_true - q[:,None], axis=0)**2

err_nearest_neighbors = np.linalg.norm(nearest_points - nearest_points_true) + np.linalg.norm(dsqs - dsqs_true)
print('err_nearest_neighbors=', err_nearest_neighbors)

#

num_queries = 11
qq = np.array(np.random.randn(K, num_queries), order='F')

all_nearest_inds, all_dsqs = KDT.nearest_neighbors_vectorized(qq, num_neighbors)

err_nearest_neighbors_vectorized = 0.0
for ii in range(num_queries):
    nearest_inds = all_nearest_inds[:,ii]
    nearest_points = pp[:, nearest_inds]
    dsqs = all_dsqs[:,ii]
    q = qq[:,ii]

    nearest_inds = np.argsort(np.linalg.norm(pp - q[:, None], axis=0))
    nearest_points_true = pp[:, nearest_inds[:num_neighbors]]
    dsqs_true = np.linalg.norm(nearest_points_true - q[:, None], axis=0) ** 2

    err_nearest_neighbors = np.linalg.norm(nearest_points - nearest_points_true) + np.linalg.norm(dsqs - dsqs_true)
    err_nearest_neighbors_vectorized += err_nearest_neighbors

print('err_nearest_neighbors_vectorized=', err_nearest_neighbors_vectorized)

# print('nearest_points=')
# print(nearest_points)
#
# print('nearest_points_true=')
# print(nearest_points_true)
#
# print('dsqs=')
# print(dsqs)
#
# actual_dsqs = np.linalg.norm(nearest_points - q[:,None], axis=0)**2
# print('actual_dsqs=')
# print(actual_dsqs)

# timing

num_pts = int(1e5)
num_queries = int(1e5)
num_neighbors = 10

pp = np.array(np.random.randn(K, num_pts), order='F')
KDT = make_KDT(pp)

qq = np.array(np.random.randn(K, num_queries), order='F')

t = time()
all_nearest_points, all_dsqs = KDT.nearest_neighbors_vectorized(qq, num_neighbors)
dt_nearest_neighbors = time() - t
print('num_pts=', num_pts, ', num_queries=', num_queries, ', num_neighbors=', num_neighbors, ', dt_nearest_neighbors=', dt_nearest_neighbors)

KDT_scipy = KDTree(np.array(pp.T, order='C'))

qqC = np.array(qq.T, order='C')

t = time()
KDT_scipy.query(qqC, k=num_neighbors)
dt_nearest_neighbors_scipy = time() - t
print('dt_nearest_neighbors_scipy=', dt_nearest_neighbors_scipy)