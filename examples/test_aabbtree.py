import numpy as np
import hlibpro_python_wrapper as hpro
from time import time
from scipy.spatial import KDTree

hcpp = hpro.hpro_cpp

K=2
num_boxes = 10

b_mins = np.random.randn(num_boxes, K)
b_maxes = np.random.randn(num_boxes, K)

AABB = hcpp.AABBTree2D(b_mins, b_maxes)





pp = np.random.randn(100,K)
KDT = make_KDT(pp)

q = np.random.randn(K)

nearest_point, dsq = KDT.nearest_neighbor(q)

nearest_ind = np.argmin(np.linalg.norm(pp - q, axis=1))
nearest_point_true = pp[nearest_ind, :]
dsq_true = np.linalg.norm(nearest_point_true - q)**2
err_nearest_one_point = np.linalg.norm(nearest_point - nearest_point_true)
err_dsq_one_point = np.abs(dsq - dsq_true)
print('err_nearest_one_point=', err_nearest_one_point)
print('err_dsq_one_point=', err_dsq_one_point)

qq = np.random.randn(77, K)
nearest_points, dsqq = KDT.nearest_neighbor_vectorized(qq)

nearest_inds = np.argmin(np.linalg.norm(pp[:,None,:] - qq[None,:,:], axis=2), axis=0)
nearest_points_true = pp[nearest_inds,:]
dsqq_true = np.linalg.norm(qq - nearest_points_true, axis=1)**2

err_nearest = np.linalg.norm(nearest_points - nearest_points_true)
print('err_nearest=', err_nearest)

err_dsqq = np.linalg.norm(dsqq - dsqq_true)
print('err_dsqq=', err_dsqq)



n_pts = int(1e6)
n_query = int(1e7)

pp = np.random.randn(n_pts, K)
t = time()
KDT = make_KDT(pp)
dt_build = time() - t
print('n_pts=', n_pts, ', dt_build=', dt_build)

qq = np.random.randn(n_query, K)
t = time()
KDT.nearest_neighbor_vectorized(qq)
dt_query = time() - t
print('n_query=', n_query, ', dt_query=', dt_query)



t = time()
KDT_scipy = KDTree(pp)
dt_build_scipy = time() - t
print('dt_build_scipy=', dt_build_scipy)

t = time()
KDT_scipy.query(qq)
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

