import numpy as np
import matplotlib.pyplot as plt
from time import time

N = 100
A = np.random.randn(N,N)
x = np.random.randn(N)
x[0] = 1e-3

y = np.dot(A, x)
A22 = A[1:, 1:]

x2 = np.linalg.solve(A22, y[1:])

err = np.linalg.norm(x2 - x[1:]) / np.linalg.norm(x[1:])
print('x[0]', x[0], ', err=', err)

#

NN = np.logspace(0,4,20, dtype=int)
dtt = list()
for N in NN:
    A = np.random.randn(N,N)
    x = np.random.randn(N)

    t = time()
    np.dot(A, x)
    dt = time() - t
    # print('N=', N, ' dt=', dt)

    dtt.append(dt)
dtt = np.array(dtt)

plt.figure()
plt.loglog(NN, dtt)
plt.xlabel('N (matrix is NxN)')
plt.ylabel('time')
plt.title('time to compute matvec A*x')

#

N = 500
nx = 10000

A = np.random.randn(N,N)
X = np.random.randn(N, nx)

t = time()
np.dot(A, X)
dt_manyvecs = time() - t
dt_avg = dt_manyvecs / nx
print('N=', N, ', dt_avg=', dt_avg)