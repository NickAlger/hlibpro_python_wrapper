import numpy as np
import matplotlib.pyplot as plt
from time import time
import hlibpro_python_wrapper as hpro

hcpp = hpro.hpro_cpp

dim = 2
num_rbf_points = 30
eval_nx = 100
eval_ny = 100

rbf_points = np.array(np.random.randn(dim, num_rbf_points), order='F')
function_at_rbf_points = np.sin(rbf_points[0,:])*np.cos(rbf_points[1,:])

xx_linear = np.linspace(-2, 2, eval_nx)
yy_linear = np.linspace(-2, 2, eval_ny)
X, Y = np.meshgrid(xx_linear, yy_linear)

eval_points = np.array([X.reshape(-1), Y.reshape(-1)], order='F')

ff = hcpp.tps_interpolate_vectorized(function_at_rbf_points,
                                     rbf_points,
                                     eval_points )

F = ff.reshape(X.shape)

plt.figure()
plt.pcolor(X, Y, F)
plt.plot(rbf_points[0,:], rbf_points[1,:], '.')
plt.colorbar()

# timing

dim = 2
num_rbf_points = 15
num_eval_points = int(1e4)

rbf_points = np.array(np.random.randn(dim, num_rbf_points), order='F')
function_at_rbf_points = np.random.randn(num_rbf_points)
eval_points = np.array(np.random.randn(dim, num_eval_points), order='F')

t = time()
hcpp.tps_interpolate_vectorized(function_at_rbf_points,
                                rbf_points,
                                eval_points )
dt = time() - t
print('dim=', dim, ', num_rbf_points=', num_rbf_points, ', num_eval_points=', num_eval_points, ', dt=', dt)
