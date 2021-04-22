import numpy as np
import matplotlib.pyplot as plt
from nalger_helper_functions import make_regular_grid
from time import time
from scipy.interpolate import interpn
import hlibpro_python_wrapper as hpw
hcpp = hpw.hpro_cpp

make_plots=False

d=2
num_pts = int(1e6)
grid_shape = (246,247)
box_min = np.array([-0.631, -0.424])
box_max = np.array([1.411, 1.105])
pp = np.random.randn(num_pts, d)
ff = np.zeros(num_pts)
(xx, yy), (X, Y) = make_regular_grid(box_min, box_max, grid_shape)
F = np.sin(2 * 2 * np.pi * np.sqrt(X**2 + Y**2))


def bilinear_interpolation(pp, box_min, box_max, F):
    grid_shape = np.array(F.shape)
    hh = (box_max - box_min) / (grid_shape - 1)
    grid_coords = (pp - box_min) / hh
    lower_inds = np.floor(grid_coords).astype(int)
    upper_inds = lower_inds + 1
    remainder = grid_coords - lower_inds
    lower_inds_mod = (lower_inds - np.floor(lower_inds / grid_shape) * grid_shape).astype(int)
    upper_inds_mod = (upper_inds - np.floor(upper_inds / grid_shape) * grid_shape).astype(int)

    s = remainder[:, 0]
    t = remainder[:, 1]

    v00 = F[lower_inds_mod[:, 0], lower_inds_mod[:, 1]]
    v01 = F[lower_inds_mod[:, 0], upper_inds_mod[:, 1]]
    v10 = F[upper_inds_mod[:, 0], lower_inds_mod[:, 1]]
    v11 = F[upper_inds_mod[:, 0], upper_inds_mod[:, 1]]

    ff = (1.0-s) * (1.0-t) * v00 + \
         (1.0-s) * t       * v01 + \
         s       * (1.0-t) * v10 + \
         s       * t       * v11

    return ff


# pp_C = pp.astype(pp.dtype, order='F')
# F_C = F.astype(F.dtype, order='F')

t = time()
ff = hcpp.bilinear_interpolation_periodic(pp, box_min, box_max, F)
dt_cpp_periodic = time() - t
print('dt_cpp_periodic=', dt_cpp_periodic)

t = time()
ff2 = interpn((xx,yy),F,pp, bounds_error=False, fill_value=0.0)
dt_scipy_fillzero = time() - t
print('dt_scipy_fillzero=', dt_scipy_fillzero)

t = time()
ff3 = hcpp.bilinear_interpolation_periodic(pp, box_min, box_max, F)
dt_numpy_periodic = time() - t
print('dt_numpy_periodic=', dt_numpy_periodic)

t = time()
ff4 = hcpp.grid_interpolate(pp, box_min[0], box_max[0], box_min[1], box_max[1], F)
dt_cpp_fillzero = time() - t
print('dt_cpp_fillzero=', dt_cpp_fillzero)

err_cpp_vs_numpy_periodic = np.linalg.norm(ff3 - ff)
print('err_cpp_vs_numpy_periodic=', err_cpp_vs_numpy_periodic)

err_cpp_vs_numpy_fillzero = np.linalg.norm(ff4 - ff2)
print('err_cpp_vs_numpy_fillzero=', err_cpp_vs_numpy_fillzero)

if make_plots:
    plt.figure()
    plt.pcolor(X, Y, F)
    plt.colorbar()

    plt.figure()
    plt.scatter(pp[:,0], pp[:,1], c=ff, s=3)
    plt.colorbar()

    plt.figure()
    plt.scatter(pp[:,0], pp[:,1], c=ff2, s=3)
    plt.colorbar()