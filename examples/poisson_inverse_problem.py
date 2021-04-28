import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import dolfin as dl
from time import time
import hlibpro_python_wrapper as hpro


#### WORK IN PROGRESS ####

########    SET UP FORWARD PROBLEM    ########

num_obs=400
a_reg = 1e0
grid_shape = (50, 55)
mesh = dl.UnitSquareMesh(*grid_shape)

V = dl.FunctionSpace(mesh, 'CG', 1)
dof_coords = V.tabulate_dof_coordinates()

u_trial = dl.TrialFunction(V)
v_test = dl.TestFunction(V)

stiffness_form = dl.inner(dl.grad(u_trial), dl.grad(v_test))*dl.dx
mass_form = u_trial*v_test*dl.dx

def top_boundary(x, on_boundary):
    return dl.near(x[0], 1.0) and on_boundary

bc = dl.DirichletBC(V, dl.Constant(0.0), top_boundary)

A = dl.assemble(stiffness_form)
bc.apply(A)

K = dl.assemble(stiffness_form)
M = dl.assemble(mass_form)
R = K + M # regularization operator

def convert_fenics_csr_matrix_to_scipy_csr_matrix(A_fenics):
    ai, aj, av = dl.as_backend_type(A_fenics).mat().getValuesCSR()
    A_scipy = sps.csr_matrix((av, aj, ai))
    return A_scipy

A_scipy = convert_fenics_csr_matrix_to_scipy_csr_matrix(A)
solve_A_scipy = spla.factorized(A_scipy)

obs_nodes = np.random.permutation(V.dim())[:num_obs]
B_scipy = sps.coo_matrix((np.ones(num_obs), (np.arange(num_obs), obs_nodes)), shape=(num_obs, V.dim())).tocsr()

R_scipy = convert_fenics_csr_matrix_to_scipy_csr_matrix(R)

def forward_map(u_numpy):
    return B_scipy * solve_A_scipy(u_numpy)

def adjoint_map(y_numpy):
    return solve_A_scipy(B_scipy.T * y_numpy, trans='T')

u_true_expr = dl.Expression('cos(12*pow(x[0],2) + 6*pow(x[1],2))', domain=mesh, degree=3)
u_true = dl.interpolate(u_true_expr, V)
u_true_numpy = u_true.vector()[:]

X = V.tabulate_dof_coordinates()
obs_pts = X[obs_nodes, :]

plt.figure()
cm = dl.plot(u_true)
plt.scatter(obs_pts[:,0], obs_pts[:,1], s=2, c='r')
plt.colorbar(cm)
plt.title('u_true and obs nodes')

y_true_numpy = forward_map(u_true_numpy)


########    OBJECTIVE, GRADIENT, HESSIAN    ########

def objective(u_numpy):
    res = y_true_numpy - forward_map(u_numpy)
    Jd = 0.5 * np.dot(res, res)
    Jr = a_reg * 0.5 * np.dot(u_numpy, R_scipy * u_numpy)
    return Jd + Jr

def gradient(u_numpy):
    res = y_true_numpy - forward_map(u_numpy)
    gd = adjoint_map(-res)
    gr = a_reg * (R_scipy * u_numpy)
    return gd + gr

def hessian_matvec(u_numpy):
    Hd_u = adjoint_map(forward_map(u_numpy))
    Hr_u = a_reg * (R_scipy * u_numpy)
    return Hd_u + Hr_u


########    FINITE DIFFERENCE CHECKS    ########

u0 = np.random.randn(V.dim())
J0 = objective(u0)
g0 = gradient(u0)

s = 1e-6
du = np.random.randn(V.dim())
u1 = u0 + s*du
J1 = objective(u1)
g1 = gradient(u1)

dJ_diff = (J1 - J0) / s
dJ = np.dot(g0, du)
err_grad = np.linalg.norm(dJ - dJ_diff) / np.linalg.norm(dJ_diff)
print('s=', s, ', err_grad=', err_grad)

dg_diff = (g1 - g0) / s
dg = hessian_matvec(du)

err_hess = np.linalg.norm(dg - dg_diff) / np.linalg.norm(dg_diff)
print('s=', s, ', err_hess=', err_hess)


########    MAKE HESSIAN H-MATRIX    ########

ct = hpro.build_cluster_tree_from_pointcloud(dof_coords, cluster_size_cutoff=50)
bct = hpro.build_block_cluster_tree(ct, ct, admissibility_eta=2.0)

A_hmatrix = hpro.build_hmatrix_from_scipy_sparse_matrix(A_scipy.tocsc(), bct)
R_hmatrix = hpro.build_hmatrix_from_scipy_sparse_matrix(R_scipy.tocsc(), bct)
BtB_hmatrix = hpro.build_hmatrix_from_scipy_sparse_matrix((B_scipy.T * B_scipy).tocsc(), bct)

iA_hmatrix = A_hmatrix.inv()

Hd_hmatrix = iA_hmatrix.T * (BtB_hmatrix * iA_hmatrix)
Hr_hmatrix = a_reg * R_hmatrix
H_hmatrix = Hd_hmatrix + Hr_hmatrix

z0 = H_hmatrix * du
z1 = hessian_matvec(du)
err_hmatrix = np.linalg.norm(z1 - z0) / np.linalg.norm(z1)
print('err_hmatrix=', err_hmatrix)


########    SOLVE INVERSE PROBLEM    ########

# iH_hmatrix = H_hmatrix.inv()
# H_hmatrix._set_symmetric()
# H_factorized_hmatrix = hpro.h_ldl(H_hmatrix) # Don't use. HPro's LDL factorization has problems
H_hmatrix._set_nonsym()
H_factorized_hmatrix = hpro.h_lu(H_hmatrix)

g = gradient(np.zeros(V.dim()))
# u_hmatrix_numpy = -iH_hmatrix * g
u_hmatrix_numpy = H_factorized_hmatrix.solve(-g)

u_hmatrix = dl.Function(V)
u_hmatrix.vector()[:] = u_hmatrix_numpy

plt.figure()
cm = dl.plot(u_hmatrix)
plt.colorbar(cm)
plt.title('u reconstructed with hmatrix')

