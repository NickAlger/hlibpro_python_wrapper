import numpy as np
from nalger_helper_functions import *
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

num_pts = 500
row_coords = np.random.randn(num_pts, 2)
col_coords = np.random.randn(num_pts, 2)

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

rr = np.random.randint(0, num_pts, num_entries)
cc = np.random.randint(0, num_pts, num_entries)

t = time()
xx = row_coords[rr,:]
yy = col_coords[cc,:]
ee_true = W1(yy)*F1(xx-yy) + W2(yy)*F2(xx-yy) + W3(yy)*F3(xx-yy)
dt_python = time() - t
print('dt_python=', dt_python)

t = time()
ee = PC_cpp.get_entries(rr,cc)
dt_cpp = time() - t
print('dt_cpp=', dt_cpp)


err = np.linalg.norm(ee_true - ee)
print('err=', err)