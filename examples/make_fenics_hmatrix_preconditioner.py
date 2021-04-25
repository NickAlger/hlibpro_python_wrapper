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


# print('sparse direct factorization')
# t = time()
# fac_A = spla.factorized(A_csc)
# dt_fac = time() - t
# print('dt_fac=', dt_fac)

########    CLUSTER TREE / BLOCK CLUSTER TREE    ########

# ct = hpro.build_cluster_tree_from_dof_coords(dof_coords, 50) # <-- still works, but not preferred
ct = hpro.build_cluster_tree_from_pointcloud(dof_coords, cluster_size_cutoff=50)
bct = hpro.build_block_cluster_tree(ct, ct, admissibility_eta=2.0)

########    BUILD HMATRIX    ########

A_hmatrix = hpro.build_hmatrix_from_scipy_sparse_matrix(A_csc, bct)

########   H-LU FACTORIZE HMATRIX    ########

iA_factorized = hpro.h_factorized_inverse(A_hmatrix, rtol=1e-1)
# iA_factorized = hpro.h_factorized_inverse(A_hmatrix, rtol=1e-1, overwrite=True) # <-- save memory, but fill A_hmatrix with nonsense

########   APPROXIMATELY SOLVE LINEAR SYSTEM    ########

x = np.random.randn(dof_coords.shape[0])
y = A_csc * x

x2 = hpro.h_factorized_solve(iA_factorized, y)

err_hfac = np.linalg.norm(x - x2)/np.linalg.norm(x2)
print('err_hfac=', err_hfac)

x3 = spla.gmres(A_csc, y, M=iA_factorized, tol=1e-12, restart=10, maxiter=1)[0]

err_gmres = np.linalg.norm(x - x3)/np.linalg.norm(x3)
print('err_gmres=', err_gmres)

#

