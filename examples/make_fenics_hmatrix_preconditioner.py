import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import fenics
from time import time
import hlibpro_python_wrapper as hpro


########    SET UP PROBLEM    ########

grid_shape = (50, 55)
mesh = fenics.UnitSquareMesh(*grid_shape)
# grid_shape = (70, 71, 72)
# mesh = fenics.UnitCubeMesh(*grid_shape)

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

A_csc = K_csc + M_csc
# A_csc = M_csc


# print('sparse direct factorization')
# t = time()
# fac_A = spla.factorized(A_csc)
# dt_fac = time() - t
# print('dt_fac=', dt_fac)


########    CLUSTER TREE / BLOCK CLUSTER TREE    ########

# ct = hpro.build_cluster_tree_from_dof_coords(dof_coords, 50) # <-- still works, but not preferred
ct = hpro.build_cluster_tree_from_pointcloud(dof_coords, cluster_size_cutoff=50)
bct = hpro.build_block_cluster_tree(ct, ct, admissibility_eta=1.0)


########    BUILD HMATRIX    ########

A_hmatrix = hpro.build_hmatrix_from_scipy_sparse_matrix(A_csc, bct)


########   H-LU FACTORIZE HMATRIX    ########

# A_factorized = hpro.h_ldl(A_hmatrix.sym(), rtol=1e-6, atol=0.0, display_progress=False, eval_type='block_wise', storage_type='store_normal')
A_factorized = hpro.h_lu(A_hmatrix, rtol=1e-6, atol=0.0, display_progress=True)


########   APPROXIMATELY SOLVE LINEAR SYSTEM    ########

x = np.random.randn(dof_coords.shape[0])
y = A_csc * x

# x2 = hpro.h_factorized_solve(iA_factorized, y)
x2 = A_factorized.solve(y)

err_hfac = np.linalg.norm(x - x2)/np.linalg.norm(x2)
print('err_hfac=', err_hfac)

x3 = spla.gmres(A_csc, y, M=A_factorized.as_linear_operator(inverse=True), tol=1e-12, restart=10, maxiter=1)[0]

err_gmres = np.linalg.norm(x - x3)/np.linalg.norm(x3)
print('err_gmres=', err_gmres)

# split

y1 = A_factorized.apply(x)

L, U = A_factorized.split()
# L, D = A_factorized.split()

y2 = L * (U * x)
# y2 = L * (D * (L.T * x))
# y2 = L * (L.T * x)

err_split = np.linalg.norm(y2 - y1) / np.linalg.norm(y1)
print('err_split=', err_split)
