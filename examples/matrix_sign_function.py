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
S = A_hmatrix.copy()
S.add_identity(s=shift, overwrite=True)
iS = S.copy()

for k in range(8):
    iS.inv(overwrite=True,rtol=1e-10)

    # norm_S = spla.eigsh(S, 1)[0][0]
    # norm_iS = spla.eigsh(iS, 1)[0][0]
    norm_S = get_largest_eigenvalue(S)
    norm_iS = get_largest_eigenvalue(iS)

    mu = np.sqrt(norm_iS / norm_S)

    print('k=', k, ', norm_S=', norm_S, ', norm_iS=', norm_iS)

    hpro.h_add(iS, S, alpha=0.5/mu, beta=0.5*mu, overwrite_B=True)
    S.copy_to(iS)

x = np.random.randn(V.dim())

np.linalg.norm(S * x - x) / np.linalg.norm(x)

A_dense = A_csc.toarray()

A_dense2 = A_dense + shift * np.eye(A_dense.shape[1])

S2 = sla.signm(A_dense2)

np.linalg.norm(S2 @ x - S * x) / np.linalg.norm(S2 @ x)

####

A_hmatrix.copy_to(iA_hmatrix)

x = np.random.randn(V.dim())

print(np.linalg.norm(A_hmatrix * x - iA_hmatrix * x))

iA_hmatrix.inv(overwrite=True)

print(np.linalg.norm(A_hmatrix * (iA_hmatrix * x) -  x) / np.linalg.norm(x))