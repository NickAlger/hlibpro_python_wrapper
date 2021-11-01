import numpy as np
import matplotlib.pyplot as plt
import hlibpro_python_wrapper as hpro
from time import time

hcpp = hpro.hpro_cpp;

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


#### RBF interpolation dropping points associated with small weights

num_pts = 100
drop_tol = 1e-4
pp = np.random.randn(num_pts, 2)

rr = np.linalg.norm(pp[None,:,:] - pp[:,None,:], axis=2)

def thin_plate_spline(rr):
    ff = np.zeros(rr.shape)
    nonzeros = (rr>0)
    ff[nonzeros] = np.power(rr[nonzeros], 2) * np.log(rr[nonzeros])
    return ff

IM = thin_plate_spline(rr)

x = np.array([[0., 0.]])
qq = thin_plate_spline(np.linalg.norm(pp - x, axis=1))
ww = np.linalg.solve(IM, qq)

good_bool = np.abs(ww) / np.max(np.abs(ww)) > drop_tol

ww_relevant = np.linalg.solve(IM[good_bool, :][:, good_bool], qq[good_bool])
ww_relevant_true = ww[good_bool]

err_relevant = np.linalg.norm(ww_relevant_true - ww_relevant) / np.linalg.norm(ww_relevant_true)
print('drop_tol=', drop_tol, ', err_relevant=', err_relevant)


#### RBF interpolation drop nodes using exact correction

bad_bool = np.logical_not(good_bool)

num_good = good_bool.sum()
num_bad = bad_bool.sum()

U = np.zeros((IM.shape[0], 2*num_bad))
U[bad_bool, :num_bad] = np.eye(num_bad)
U[good_bool, num_bad:] = IM[np.ix_(good_bool, bad_bool)]

# V = np.zeros((2*num_bad, IM.shape[1]))
# U[bad_bool, :num_bad] = np.eye(num_bad)
# U[good_bool, num_bad:] = IM[np.ix_(good_bool, bad_bool)]

#

A = np.random.randn(100,137)

bad_rows = np.zeros(A.shape[0], dtype=bool)
bad_rows[10:15] = True

bad_cols = np.zeros(A.shape[1], dtype=bool)
bad_cols[22:38] = True

# bad_rows = (A[:,0] > 0)
# bad_cols = (A[0,:] > 0)

good_rows = np.logical_not(bad_rows)
good_cols = np.logical_not(bad_cols)

n_bad_rows = bad_rows.sum()
n_bad_cols = bad_cols.sum()
n_good_rows = good_rows.sum()
n_good_cols = good_cols.sum()

U1 = np.zeros((A.shape[0], n_bad_rows))
U1[bad_rows, :] = np.eye(n_bad_rows)

U2 = np.zeros((A.shape[0], n_bad_cols))
U2[good_rows, :] = A[np.ix_(good_rows, bad_cols)]

U = np.hstack([U1, U2])

V1 = np.zeros((n_bad_cols, A.shape[1]))
V1[:, bad_cols] = np.eye(n_bad_cols)

V2 = np.zeros((n_bad_rows, A.shape[1]))
V2[:, good_cols] = A[np.ix_(bad_rows, good_cols)]

V = np.vstack([V2, V1])

dA = np.dot(U, V)
plt.matshow(dA)

B = A - np.dot(U, V)
B_true = A.copy()
B_true[:, bad_cols] = 0.0
B_true[bad_rows, :] = 0.0
B_true[np.ix_(bad_rows, bad_cols)] = A[np.ix_(bad_rows, bad_cols)]

err = np.linalg.norm(B - B_true) / np.linalg.norm(B_true)
print('err=', err)


def submatrix_deletion_factors(A, bad_rows, bad_cols):
    good_rows = np.setdiff1d(np.arange(A.shape[0]), bad_rows)
    good_cols = np.setdiff1d(np.arange(A.shape[1]), bad_cols)

    U = np.zeros((A.shape[0], len(bad_rows) + len(bad_cols)))
    U[bad_rows, :len(bad_rows)] = np.eye(len(bad_rows))
    U[good_rows, len(bad_rows):] = A[np.ix_(good_rows, bad_cols)]

    V = np.zeros((len(bad_rows) + len(bad_cols), A.shape[1]))
    V[:len(bad_rows), good_cols] = A[np.ix_(bad_rows, good_cols)]
    V[len(bad_rows):, bad_cols] = np.eye(len(bad_cols))

    return U, V


A = np.random.randn(102,157)

bad_rows = [1, 2, 37, 15, 6, 7, 8]
bad_cols = [80, 60, 40]

U, V = submatrix_deletion_factors(A, bad_rows, bad_cols)

B = A - np.dot(U, V)
B_true = A.copy()
B_true[:, bad_cols] = 0.0
B_true[bad_rows, :] = 0.0
B_true[np.ix_(bad_rows, bad_cols)] = A[np.ix_(bad_rows, bad_cols)]

err = np.linalg.norm(B - B_true) / np.linalg.norm(B_true)
print('err=', err)

good_rows = np.setdiff1d(np.arange(A.shape[0]), bad_rows)
good_cols = np.setdiff1d(np.arange(A.shape[1]), bad_cols)

U2, V2 = hcpp.submatrix_deletion_factors(A, bad_rows, bad_cols)
V2 = -V2

err_deletion_factors_cpp = np.linalg.norm(U - U2) + np.linalg.norm(V - V2)
print('err_deletion_factors_cpp=', err_deletion_factors_cpp)


def submatrix_woodbury_solve(B_good, solve_A, bad_rows, bad_cols):
    # Solves A[good_rows, good_cols] * X[good_cols,:] = B_good[good_rows,:]
    #   A.shape = (N, N)
    #   B.shape = X.shape = (N, k)
    #   len(bad_rows) = len(bad_cols) = b < N
    #   P = solve_A(Q) solves A*P = Q, where P.shape = Q.shape = (N,k)
    # Uses Woodbury formula:
    #   inv(A - UV) = inv(A) + inv(A)*U*inv(I - V^T*inv(A)*U)*V*inv(A)
    U, V = submatrix_deletion_factors(A, bad_rows, bad_cols)
    Z = solve_A(U) # Z = inv(A)*U
    C = np.eye(2*len(bad_rows)) - np.dot(V, Z) # C = I - V*inv(A)*U

    good_rows = np.setdiff1d(np.arange(A.shape[0]), bad_rows)
    good_cols = np.setdiff1d(np.arange(A.shape[1]), bad_cols)

    B = np.zeros((A.shape[0], B_good.shape[1]))
    B[good_rows, :] = B_good

    X = solve_A(B) # x0 = inv(A)*b
    X += np.dot(Z, np.linalg.solve(C, np.dot(V, X))) # dx = inv(A)*U*inv(C)*V*x0
    return X[good_cols, :]


A = np.random.randn(102,102)

bad_rows = [2,   1, 37, 15]
bad_cols = [80, 60, 40, 55]

good_rows = np.setdiff1d(np.arange(A.shape[0]), bad_rows)
good_cols = np.setdiff1d(np.arange(A.shape[1]), bad_cols)

X = np.random.randn(A.shape[0], 13)
B_good = np.dot(A[np.ix_(good_rows, good_cols)], X[good_cols, :])
solve_A = lambda Z: np.linalg.solve(A, Z)

X_good = submatrix_woodbury_solve(B_good, solve_A, bad_rows, bad_cols)
err = np.linalg.norm(X[good_cols,:] - X_good) / np.linalg.norm(X[good_cols,:])
print('err=', err)

#

import numpy as np

N = 102
A = np.random.randn(N, N)

bad_rows = [2,   1, 37, 15]
bad_cols = [80, 60, 40, 55]
k = len(bad_rows)

good_rows = np.setdiff1d(np.arange(A.shape[0]), bad_rows)
good_cols = np.setdiff1d(np.arange(A.shape[1]), bad_cols)

A_tilde_true = A.copy()
A_tilde_true[:, bad_cols] = 0.0
A_tilde_true[bad_rows, :] = 0.0
A_tilde_true[np.ix_(bad_rows, bad_cols)] = A[np.ix_(bad_rows, bad_cols)]

U = np.zeros((A.shape[0], 2*k))
U[bad_rows, :k] = np.eye(k)
U[good_rows, k:] = A[np.ix_(good_rows, bad_cols)]

V = np.zeros((2*k, A.shape[1]))
V[:k, good_cols] = A[np.ix_(bad_rows, good_cols)]
V[k:, bad_cols] = np.eye(k)

A_tilde = A - np.dot(U, V)
err_UV = np.linalg.norm(A_tilde_true - A_tilde) / np.linalg.norm(A_tilde_true)
print('err_UV=', err_UV)


# solve linear system involving submatrix

b = np.random.randn(N-k)
x_true = np.linalg.solve(A[np.ix_(good_rows, good_cols)], b)

Z = np.linalg.solve(A, U) # Z = inv(A)*U
C = np.eye(2*k) - np.dot(V, Z) # C = I - V*inv(A)*U

b_tilde = np.zeros(N)
b_tilde[good_rows] = b

x_tilde = np.linalg.solve(A_tilde, b_tilde) # x0 = inv(A)*b
x_tilde += np.dot(Z, np.linalg.solve(C, np.dot(V, x_tilde))) # dx = inv(A)*U*inv(C)*V*x0
x = x_tilde[good_cols]

err_woodbury = np.linalg.norm(x - x_true) / np.linalg.norm(x_true)
print('err_woodbury=', err_woodbury)

#

A = np.random.randn(7,7)
U = np.random.randn(7,3)
V = np.random.randn(3,7)
b = np.random.randn(7)

iA = np.linalg.inv(A)
x = np.dot(iA, b)
hcpp.woodbury_update(x, A, iA, U, V)

err_woodbury_cpp = np.linalg.norm(np.dot(A + np.dot(U, V), x) - b) / np.linalg.norm(b)
print('err_woodbury_cpp=', err_woodbury_cpp)

