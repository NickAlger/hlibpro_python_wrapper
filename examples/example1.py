import numpy as np
import scipy.sparse as sps
import fenics
import hlibpro_python_wrapper as hpro


########    SETUP    ########

mesh = fenics.UnitSquareMesh(70, 75)
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

# #### Test building objects within a function (this used to cause memory errors segfaults) ####
#
# def asdf():
#     ct = hpro.build_cluster_tree_from_dof_coords(dof_coords, cluster_size_cutoff=50)
#     bct = hpro.build_block_cluster_tree(ct, ct, admissibility_eta=2.0)
#     K_hmatrix = hpro.build_hmatrix_from_scipy_sparse_matrix(K_csc, bct)
#     return K_hmatrix
#
# K_hmatrix = asdf()
#
# x = np.random.randn(dof_coords.shape[0])
# y = hpro.h_matvec(K_hmatrix, x)

########    CLUSTER TREE / BLOCK CLUSTER TREE    ########

ct = hpro.build_cluster_tree_from_pointcloud(dof_coords, cluster_size_cutoff=50)
bct = hpro.build_block_cluster_tree(ct, ct, admissibility_eta=2.0)
ct.visualize("sparsemat_cluster_tree_from_python")
bct.visualize("sparsemat_block_cluster_tree_from_python")

########    BUILD HMATRIX FROM SPARSE    ########

K_hmatrix = hpro.build_hmatrix_from_scipy_sparse_matrix(K_csc, bct)
M_hmatrix = hpro.build_hmatrix_from_scipy_sparse_matrix(M_csc, bct)

K_hmatrix.visualize("stiffness_hmatrix_from_python")


x = np.random.randn(dof_coords.shape[0])
y = K_hmatrix * x  # alternatively: hpro.h_matvec(K_hmatrix, x)
y2 = K_csc * x

err_hmat_stiffness = np.linalg.norm(y - y2)/np.linalg.norm(y2)
print('err_hmat_stiffness=', err_hmat_stiffness)


########    SCALE HMATRIX   ########

three_K_hmatrix = 3.0 * K_hmatrix  # alternative: three_K_hmatrix = hpro.h_scale(K_hmatrix, 3.0)
three_y = three_K_hmatrix * x      # alternative: three_y = hpro.h_matvec(three_K_hmatrix, x)

err_h_scale = np.linalg.norm(three_y - 3.0*y)
print('err_h_scale=', err_h_scale)


########    VISUALIZE HMATRIX    ########

M_hmatrix.visualize("mass_hmatrix_from_python") # hpro.visualize_hmatrix(M_hmatrix, "mass_hmatrix_from_python")

x = np.random.randn(dof_coords.shape[0])
y = M_hmatrix * x
y2 = M_csc * x

err_hmat_mass = np.linalg.norm(y - y2)/np.linalg.norm(y2)
print('err_hmat_mass=', err_hmat_mass)


########    ADD HMATRICES   ########

A_hmatrix = hpro.h_add(K_hmatrix, M_hmatrix)  # alternatively: K_hmatrix + M_hmatrix

x = np.random.randn(dof_coords.shape[0])
y = hpro.h_matvec(A_hmatrix, x)
y2 = A_csc * x

err_h_add = np.linalg.norm(y - y2)/np.linalg.norm(y2)
print('err_h_add=', err_h_add)


########    MULTIPLY HMATRICES   ########

KM_hmatrix = hpro.h_mul(K_hmatrix, M_hmatrix, rtol=1e-6)  # alternatively: K_hmatrix * M_hmatrix

x = np.random.randn(dof_coords.shape[0])
y = hpro.h_matvec(KM_hmatrix, x)

y2 = hpro.h_matvec(K_hmatrix, hpro.h_matvec(M_hmatrix, x))
# y4 = 0.5 * (K_csc * (M_csc * x) + M_csc * (K_csc * x))
err_H_mul = np.linalg.norm(y2 - y)/np.linalg.norm(y2)
print('err_H_mul=', err_H_mul)

y3 = K_csc * (M_csc * x)
err_H_mul_exact = np.linalg.norm(y3 - y)/np.linalg.norm(y2)
print('err_H_mul_exact=', err_H_mul_exact)


########    FACTORIZE HMATRIX   ########

A_factorized = hpro.h_lu(A_hmatrix, rtol=1e-10)
# iA_factorized = hpro.h_factorized_inverse(A_hmatrix, rtol=1e-10)

x = np.random.randn(dof_coords.shape[0])
y = hpro.h_matvec(A_hmatrix, x)

x2 = A_factorized.solve(y)
# x2 = hpro.h_factorized_solve(iA_factorized, y)

err_hfac = np.linalg.norm(x - x2)/np.linalg.norm(x2)
print('err_hfac=', err_hfac)

A_factorized.visualize('inv_A_factors')
# hpro.visualize_inverse_factors(iA_factorized, 'inv_A_factors')


########    MULTIPLY HMATRIX BY DIAGONAL MATRIX   ########

A2_hmatrix = A_hmatrix.copy()

# left

dd = np.random.randn(A2_hmatrix.shape[0])
A2_hmatrix.mul_diag_left(dd)

z = np.random.randn(A2_hmatrix.shape[1])

y1 = dd * (A_hmatrix * z)
y2 = A2_hmatrix * z

err_mul_diag_left = np.linalg.norm(y2 - y1) / np.linalg.norm(y1)
print('err_mul_diag_left=', err_mul_diag_left)

# right

A3_hmatrix = A_hmatrix.copy()

dd = np.random.randn(A3_hmatrix.shape[0])
A3_hmatrix.mul_diag_right(dd)

z = np.random.randn(A3_hmatrix.shape[1])

y3 = A_hmatrix * (dd * z)
y4 = A3_hmatrix * z

err_mul_diag_right = np.linalg.norm(y3 - y4) / np.linalg.norm(y3)
print('err_mul_diag_right=', err_mul_diag_right)


########    LOW RANK UPDATE   ########

rank = 13
U = np.random.randn(A_hmatrix.shape[0], rank)
V = np.random.randn(rank, A_hmatrix.shape[1])

z = np.random.randn(A_hmatrix.shape[1])
q1 = A_hmatrix * z + np.dot(U, np.dot(V, z))

A_plus_UV_hmatrix = A_hmatrix.low_rank_update(U, V)

q2 = A_plus_UV_hmatrix * z

err_low_rank_update = np.linalg.norm(q1 - q2) / np.linalg.norm(q1)
print('err_low_rank_update=', err_low_rank_update)


########    SYMMETRIC POSITIVE DEFINITE MODIFICATION    ########

shift = -5e-2
cutoff = 0.0

A_indefinite_hmatrix = A_hmatrix.add_identity(s=shift)

A_plus_hmatrix = A_indefinite_hmatrix.spd(cutoff=cutoff)

A_indefinite_dense = A_csc.toarray() + shift * np.eye(A_csc.shape[0])

print('computing spectral decomposition via dense brute force to check spd modification')
dd, P = np.linalg.eigh(A_indefinite_dense)

A_plus_dense = np.dot(P, np.dot(np.diag(np.abs(dd)), P.T))

z = np.random.randn(A_plus_dense.shape[1])

q1 = np.dot(A_plus_dense, z)
q2 = A_plus_hmatrix * z

q1b = A_csc * z
discrepancy_pre_spd = np.linalg.norm(q2-q1b)/np.linalg.norm(q1b)
print('discrepancy_pre_spd=', discrepancy_pre_spd)

err_spd = np.linalg.norm(q2-q1)/np.linalg.norm(q1)
print('err_spd=', err_spd)


########    DFP UPDATE    ########

B_hmatrix = A_hmatrix.low_rank_update(U, U.T).add_identity(s=0.1)

X = np.random.randn(A_hmatrix.shape[0], 19)
Y = np.zeros((A_hmatrix.shape[0], X.shape[1]))
for k in range(X.shape[1]):
    Y[:,k] = B_hmatrix * X[:,k]

A_hmatrix_dfp = A_hmatrix.dfp_update(X, Y, rtol=1e-12, atol=1e-12)

A_hmatrix_dfp.visualize('A_hmatrix_dfp')

iA_dfp = A_hmatrix_dfp.inv(atol=1e-3, rtol=1e-3)

iA_dfp.visualize('iA_hmatrix_dfp')