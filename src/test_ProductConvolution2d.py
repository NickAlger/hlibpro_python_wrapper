import numpy as np
from nalger_helper_functions import *
import matplotlib.pyplot as plt
from time import time
from hlibpro_python_wrapper import *

W1 = BoxFunction(np.array([-1.1, -0.9]), np.array([0.5, 0.55]), np.zeros((21,15)))
XW1, YW1 = W1.meshgrid
W1.array = np.cos(XW1)
W1.plot(title='W1')

W2 = BoxFunction(np.array([-0.53, -0.52]), np.array([1.1, 1.0]), np.zeros((17,18)))
XW2, YW2 = W2.meshgrid
W2.array = np.cos(YW2)
W2.plot(title='W2')

W3 = BoxFunction(np.array([-1.0, -0.25]), np.array([0.4, 0.9]), np.zeros((14,16)))
XW3, YW3 = W3.meshgrid
W3.array = np.cos(YW3 + XW3)
W3.plot(title='W3')

F1 = BoxFunction(np.array([-1.0, -0.9]), np.array([0.8, 0.85]), np.zeros((19,17)))
XF1, YF1 = F1.meshgrid
F1.array = np.exp(-0.5 * (XF1**2 + YF1**2) / (0.25)**2)
F1.plot(title='F1')

F2 = BoxFunction(np.array([-0.85, -0.95]), np.array([1.1, 1.2]), np.zeros((23,20)))
XF2, YF2 = F2.meshgrid
F2.array = np.exp(-0.5 * (XF2**2 + YF2**2) / (0.35)**2) * np.cos(6 * np.sqrt(((XF2+YF2)**2 + YF2**2)))
F2.plot(title='F2')

F3 = BoxFunction(np.array([-1.03, -1.04]), np.array([1.05, 1.06]), np.zeros((35,21)))
XF3, YF3 = F3.meshgrid
F3.array = np.exp(-0.5 * ((XF3+0.1)**2 + (YF3-0.15)**2) / (0.2)**2) * np.cos(8 * XF3)
F3.plot(title='F3')

num_row_pts = 511
# num_col_pts = 709
num_col_pts = num_row_pts
row_coords = np.random.randn(num_row_pts, 2)
# col_coords = np.random.randn(num_col_pts, 2)
col_coords = row_coords

WW_mins = [W1.min, W2.min, W3.min]
WW_maxes = [W1.max, W2.max, W3.max]
WW_arrays = [W1.array, W2.array, W3.array]

FF_mins = [F1.min, F2.min, F3.min]
FF_maxes = [F1.max, F2.max, F3.max]
FF_arrays = [F1.array, F2.array, F3.array]

PC_cpp = hpro_cpp.ProductConvolution2d(WW_mins, WW_maxes, WW_arrays,
                                       FF_mins, FF_maxes, FF_arrays,
                                       row_coords, col_coords)

num_entries = 53

rr = np.random.randint(0, num_row_pts, num_entries)
cc = np.random.randint(0, num_col_pts, num_entries)

t = time()
yy = row_coords[rr, :]
xx = col_coords[cc, :]
ee_true = W1(xx) * F1(yy - xx) + W2(xx) * F2(yy - xx) + W3(xx) * F3(yy - xx)
dt_get_entries_python = time() - t
print('dt_get_entries_python=', dt_get_entries_python)

t = time()
ee = PC_cpp.get_entries(rr,cc)
dt_get_entries_cpp = time() - t
print('dt_get_entries_cpp=', dt_get_entries_cpp)

err_get_entries = np.linalg.norm(ee_true - ee)
print('err_get_entries=', err_get_entries)

num_rows_block = 54
num_cols_block = 81

block_rows = np.random.randint(0, num_row_pts, num_rows_block)
block_cols = np.random.randint(0, num_col_pts, num_cols_block)

t = time()
B = PC_cpp.get_block(block_rows, block_cols)
dt_get_block_cpp = time() - t
print('dt_get_block_cpp=', dt_get_block_cpp)

t = time()
R = np.outer(block_rows, np.ones(num_cols_block, dtype=int)).reshape(-1)
C = np.outer(np.ones(num_rows_block, dtype=int), block_cols).reshape(-1)
rr_block = R.reshape(-1)
cc_block = C.reshape(-1)
xx_block = row_coords[rr_block,:]
yy_block = col_coords[cc_block,:]
B_true = (W1(yy_block) * F1(xx_block - yy_block) +
          W2(yy_block) * F2(xx_block - yy_block) +
          W3(yy_block) * F3(xx_block - yy_block)).reshape(num_rows_block, num_cols_block)
dt_get_block_python = time() - t
print('dt_get_block_python=', dt_get_block_python)

err_get_block = np.linalg.norm(B_true - B)
print('err_get_block=', err_get_block)

t = time()
M = PC_cpp.get_array()
dt_get_array_cpp = time() - t
print('dt_get_array_cpp=', dt_get_array_cpp)

num_checks = 94
rr_check = np.random.randint(0, num_row_pts, num_checks)
cc_check = np.random.randint(0, num_col_pts, num_checks)
errs_get_array = np.inf * np.ones(num_checks)
for k in range(num_checks):
    r = rr_check[k]
    c = cc_check[k]
    y = row_coords[r,:].reshape((1,-1))
    x = col_coords[c,:].reshape((1,-1))
    entry = M[r,c]
    entry_true = W1(x) * F1(y - x) + W2(x) * F2(y - x) + W3(x) * F3(y - x)
    errs_get_array[k] = np.abs(entry - entry_true)

err_get_array = np.linalg.norm(errs_get_array)
print('err_get_array=', err_get_array)

##

u = np.random.randn(M.shape[1])

tol=1e-6

row_ct = build_cluster_tree_from_dof_coords(row_coords, 60)
col_ct = build_cluster_tree_from_dof_coords(col_coords, 60)
bct = build_block_cluster_tree(row_ct, col_ct, 1.0)

M_hmatrix = build_product_convolution_hmatrix_2d(WW_mins, WW_maxes, WW_arrays,
                                                 FF_mins, FF_maxes, FF_arrays,
                                                 row_coords, col_coords, bct, tol=tol)

v1 = np.dot(M, u)
v2 = M_hmatrix.matvec(u)

err_hmatrix = np.linalg.norm(v2 - v1) / np.linalg.norm(v1)
print('err_hmatrix=', err_hmatrix)
