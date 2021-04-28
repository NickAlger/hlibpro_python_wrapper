import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import scipy.linalg as sla
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

def get_largest_eigenvalue(H_hmatrix, max_iter=50, tol=1e-2, display=False):
    u = np.random.randn(H_hmatrix.shape[1])
    u = u / np.linalg.norm(u)
    eig = 0.0
    for k in range(max_iter):
        u2 = H_hmatrix * u
        eig2 = np.linalg.norm(u2)
        u2 = u2 / eig2
        diff = np.linalg.norm(u2 - u)
        # diff = np.abs(eig2 - eig)
        if display:
            print('k=', k, ', eig2=', eig2, ', diff=', diff)
        u = u2
        eig = eig2
        if diff < tol:
            break
    return eig

eig = get_largest_eigenvalue(A_hmatrix, display=True)
eig_true = spla.eigsh(A_hmatrix, 1)[0][0]
print('eig=', eig, ', eig_true=', eig_true)


shift = -2.34567

inv_tol = 1e-9
newton_tol = 1e-4

M1 = A_hmatrix.copy()
M1.add_identity(s=shift, overwrite=True)
M2 = M1.copy()
# M1 = M2 = S_old

norm_S = get_largest_eigenvalue(M1)

for k in range(7):
    M2.inv(overwrite=True, rtol=inv_tol)
    # M1 = S_old, M2 = inv(S_old),

    norm_iS = get_largest_eigenvalue(M2)
    mu = np.sqrt(norm_iS / norm_S)
    print('k=', k, ', norm_S=', norm_S, ', norm_iS=', norm_iS)

    hpro.h_add(M1, M2, alpha=0.5 * mu, beta=0.5 / mu, overwrite_B=True)
    # M1 = S_old, M2 = S_new

    norm_S = get_largest_eigenvalue(M2)

    hpro.h_add(M2, M1, alpha=1.0, beta=-1.0, overwrite_B=True)
    # M1 = S_new - S_old, M2 = S_new

    norm_diff = get_largest_eigenvalue(M1)
    rel_err = norm_diff / norm_S
    print('rel_err=', rel_err)

    M2.copy_to(M1)
    # M1 = M2 = S_new

    if rel_err**2 < newton_tol:
        break

S = M1

x = np.random.randn(V.dim())

np.linalg.norm(S * x - x) / np.linalg.norm(x)

A_dense = A_csc.toarray()

A_dense2 = A_dense + shift * np.eye(A_dense.shape[1])

S2 = sla.signm(A_dense2)

np.linalg.norm(S2 @ x - S * x) / np.linalg.norm(S2 @ x)

####
