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


########    CLUSTER TREE / BLOCK CLUSTER TREE    ########

ct = hpro.build_cluster_tree_from_pointcloud(dof_coords, cluster_size_cutoff=50)
bct = hpro.build_block_cluster_tree(ct, ct, admissibility_eta=1.0)


########    BUILD HMATRIX    ########

A_hmatrix = hpro.build_hmatrix_from_scipy_sparse_matrix(A_csc, bct)


########    COMPUTE SIGN FUNCTION    ########

iA_hmatrix = A_hmatrix.inv()

A_hmatrix.copy_to(iA_hmatrix)

x = np.random.randn(V.dim())

print(np.linalg.norm(A_hmatrix * x - iA_hmatrix * x))

iA_hmatrix.inv(overwrite=True)

print(np.linalg.norm(A_hmatrix * (iA_hmatrix * x) -  x) / np.linalg.norm(x))