import numpy as np
import scipy.linalg as sla
import scipy.sparse as sps
import matplotlib.pyplot as plt
import hlibpro_python_wrapper as hpro

np.random.seed(0)
N = 1000
B_csc = sps.diags([-np.ones(N-1), 2.0*np.ones(N), -1.0*np.ones(N-1)], offsets=[-1,0,1]).tocsc()

iA_offdiag = np.random.randn(N-1)
iA_diag = np.random.randn(N) + 1.25*np.ones(N)
iA_csc = sps.diags([iA_offdiag, iA_diag, iA_offdiag], offsets=[-1,0,1]).tocsc()

pp = np.linspace(-1.0, 1.0, N).reshape((-1,1))

ct = hpro.build_cluster_tree_from_pointcloud(pp)
bct = hpro.build_block_cluster_tree(ct, ct)

B_hmatrix = hpro.build_hmatrix_from_scipy_sparse_matrix(B_csc, bct)

iA_hmatrix = hpro.build_hmatrix_from_scipy_sparse_matrix(iA_csc, bct)
A_hmatrix = iA_hmatrix.inv()

threshold = -0.25

dd, V, shifts, factorized_shifted_matrices, LM_eig = hpro.negative_eigenvalues_of_hmatrix_pencil(A_hmatrix, B_hmatrix, threshold=threshold)

iA_dense = iA_csc.toarray()
B_dense = B_csc.toarray()

A_dense = np.linalg.inv(iA_dense)

ee, P = sla.eigh(A_dense, B_dense)

num_positive_eigs = np.sum(ee > 0)
print('num_positive_eigs / N =', num_positive_eigs, ' / ', N)

ee_bad = ee[ee<threshold]
ee_bad_approx = np.sort(dd)[:len(ee_bad)]

dd_err = np.linalg.norm(ee_bad - ee_bad_approx) / np.linalg.norm(ee_bad)
print('dd_err=', dd_err)

A_plus_dense = A_dense - V @ np.diag(dd) @ V.T

rayleigh_matrix = P.T @ A_plus_dense @ P
rayleigh = rayleigh_matrix.diagonal()
rayleigh_offdiag_err = np.linalg.norm(rayleigh_matrix - np.diag(rayleigh)) / np.linalg.norm(rayleigh_matrix)
print('rayleigh_offdiag_err=', rayleigh_offdiag_err)

negative_eig_err = np.linalg.norm(rayleigh[ee < threshold]) / np.linalg.norm(rayleigh)
print('negative_eig_err=', negative_eig_err)

positive_eig_err = np.linalg.norm(rayleigh[ee >= 0.0] - ee[ee >= 0.0]) / np.linalg.norm(ee[ee >= 0.0])
print('positive_eig_err=', positive_eig_err)

intermediate_big_enough = np.all(threshold < rayleigh[(threshold < ee) * (ee < 0.0)])
print('intermediate_big_enough=', intermediate_big_enough)

intermediate_small_enough = np.all(rayleigh[(threshold < ee) * (ee < 0.0)] < 1e-8)
print('intermediate_small_enough=', intermediate_small_enough)

LM_true = ee[np.argmax(np.abs(ee))]
LM_eig_err = np.abs(LM_true - LM_eig) / np.abs(LM_true)
print('LM_eig_err=', LM_eig_err)

b = np.random.randn(N)
for mu, fac in zip(shifts, factorized_shifted_matrices):
    x_true = np.linalg.solve(A_dense - mu*B_dense, b)
    x = fac.matvec(b)
    fac_err = np.linalg.norm(x - x_true) / np.linalg.norm(x_true)
    print('mu=', mu, ', fac_err=', fac_err)

##

mu_max = -np.min(ee)*1.0
mu_min = 0.5

shifted_interpolator = hpro.make_shifted_hmatrix_inverse_interpolator(
    A_hmatrix, B_hmatrix, mu_min, mu_max, gamma=-1.0,
    mu_spacing_factor=50.0,
    deflation_dd=dd, deflation_V=V, LM_eig=LM_eig, known_mus=[-s for s in shifts],
    known_shifted_factorizations=factorized_shifted_matrices)


b = np.random.randn(N)
mus = list(np.logspace(np.log10(mu_min), np.log10(mu_max), 100))
# mus = shifted_interpolator.known_mus
errs = []
for mu in mus:
    x_true = np.linalg.solve(A_plus_dense + mu * B_dense, b)
    x = shifted_interpolator.solve_shifted_deflated_preconditioner(b, mu, display=False)
    # x = shifted_interpolator.solve_shifted_deflated(b, mu, display=False)
    err = np.linalg.norm(x_true - x) / np.linalg.norm(x_true)
    errs.append(err)
    print('mu=', mu, ', err=', err)

plt.loglog(mus, errs)