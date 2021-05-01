import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import scipy.linalg as sla
import fenics
from time import time
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import hlibpro_python_wrapper as hpro


########    SET UP PROBLEM    ########

grid_shape = (50, 55)
mesh = fenics.UnitSquareMesh(*grid_shape)

V = fenics.FunctionSpace(mesh, 'CG', 1)
dof_coords = V.tabulate_dof_coordinates()

u_trial = fenics.TrialFunction(V)
v_test = fenics.TestFunction(V)

stiffness_form = fenics.inner(fenics.grad(u_trial), fenics.grad(v_test))*fenics.dx
mass_form = u_trial*v_test*fenics.dx

K = fenics.assemble(stiffness_form)
M = fenics.assemble(mass_form)

def convert_fenics_csr_matrix_to_scipy_csr_matrix(A_fenics):
    ai, aj, av = fenics.as_backend_type(A_fenics).mat().getValuesCSR()
    A_scipy = sps.csr_matrix((av, aj, ai))
    return A_scipy

K_csc = convert_fenics_csr_matrix_to_scipy_csr_matrix(K).tocsc()
M_csc = convert_fenics_csr_matrix_to_scipy_csr_matrix(M).tocsc()
I_csc = sps.eye(V.dim()).tocsc()

A_csc = K_csc + M_csc -1.2345 * I_csc  # negative shift to make matrix indefinite


########    BUILD HMATRIX    ########

ct = hpro.build_cluster_tree_from_pointcloud(dof_coords, cluster_size_cutoff=50)
bct = hpro.build_block_cluster_tree(ct, ct, admissibility_eta=2.0)

A_hmatrix = hpro.build_hmatrix_from_scipy_sparse_matrix(A_csc, bct)


########    COMPUTE HMATRIX SPD APPROXIMATION, METHOD 1    ########

A_spd_hmatrix = hpro.rational_positive_definite_approximation_method1(A_hmatrix, overwrite=False, atol_inv=1e-6)

A_spd_hmatrix.visualize('A_spd_hmatrix')


########    CHECK CORRECTNESS VIA DENSE LINEAR ALGEBRA, METHOD 1    ########

A_dense = np.zeros(A_hmatrix.shape)
for k in tqdm(range(A_dense.shape[1])):
    ek = np.zeros(A_dense.shape[1])
    ek[k] = 1.0
    A_dense[:,k] = A_hmatrix * ek

A_spd_dense = np.zeros(A_spd_hmatrix.shape)
for k in tqdm(range(A_spd_dense.shape[1])):
    ek = np.zeros(A_spd_dense.shape[1])
    ek[k] = 1.0
    A_spd_dense[:,k] = A_spd_hmatrix * ek

dd, P = np.linalg.eigh(A_dense)
_, P_spd = np.linalg.eigh(A_spd_dense)
RQ = np.dot(P.T, np.dot(A_spd_dense, P))
dd_spd = RQ.diagonal()

A_dense2 = np.dot(P_spd, np.dot(np.diag(dd), P_spd.T))

err_eigenvectors1 = np.linalg.norm(A_dense - A_dense) / np.linalg.norm(A_dense)
print('err_eigenvectors1=', err_eigenvectors1)

err_eigenvectors2 = np.linalg.norm(RQ - np.diag(RQ.diagonal())) / np.linalg.norm(RQ)
print('err_eigenvectors2=', err_eigenvectors2)

plt.figure()
plt.plot(dd)
plt.plot(dd_spd)
plt.xlabel('i')
plt.ylabel('lambda_i')
plt.legend(['Original A', 'SPD approximation of A'])
plt.title('SPD eigenvalue modifications, method 1')


########    COMPUTE HMATRIX SPD APPROXIMATION, METHOD 2    ########

A_spd_hmatrix = hpro.rational_positive_definite_approximation_method2(A_hmatrix, overwrite=False, atol_inv=1e-6)

A_spd_hmatrix.visualize('A_spd_hmatrix')


########    CHECK CORRECTNESS VIA DENSE LINEAR ALGEBRA, METHOD 2    ########

A_dense = np.zeros(A_hmatrix.shape)
for k in tqdm(range(A_dense.shape[1])):
    ek = np.zeros(A_dense.shape[1])
    ek[k] = 1.0
    A_dense[:,k] = A_hmatrix * ek

A_spd_dense = np.zeros(A_spd_hmatrix.shape)
for k in tqdm(range(A_spd_dense.shape[1])):
    ek = np.zeros(A_spd_dense.shape[1])
    ek[k] = 1.0
    A_spd_dense[:,k] = A_spd_hmatrix * ek

dd, P = np.linalg.eigh(A_dense)
_, P_spd = np.linalg.eigh(A_spd_dense)
RQ = np.dot(P.T, np.dot(A_spd_dense, P))
dd_spd = RQ.diagonal()

A_dense2 = np.dot(P_spd, np.dot(np.diag(dd), P_spd.T))

err_eigenvectors1 = np.linalg.norm(A_dense - A_dense) / np.linalg.norm(A_dense)
print('err_eigenvectors1=', err_eigenvectors1)

err_eigenvectors2 = np.linalg.norm(RQ - np.diag(RQ.diagonal())) / np.linalg.norm(RQ)
print('err_eigenvectors2=', err_eigenvectors2)

plt.figure()
plt.plot(dd)
plt.plot(dd_spd)
plt.xlabel('i')
plt.ylabel('lambda_i')
plt.legend(['Original A', 'SPD approximation of A'])
plt.title('SPD eigenvalue modifications, method 2')


