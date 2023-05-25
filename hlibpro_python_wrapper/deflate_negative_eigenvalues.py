import numpy as np
import scipy.sparse.linalg as spla
import typing as typ
from dataclasses import dataclass
from functools import cached_property


_GMRES_TOL = 1e-10

vec2vec = typ.Callable[[np.ndarray], np.ndarray]

def solve_shifted_deflated(
        b: np.ndarray,
        solve_A_minus_sigma_B: vec2vec,
        sigma: float,
        gamma: float,
        dd: np.ndarray,
        BU: np.ndarray
) -> np.ndarray:
    '''solves (A - sigma*B + gamma * (B @ U) @ diag(dd) @ (B @ U).T) @ x = b
    A must be symmetric
    B must be symmetric positive definite
    dd, U must be generalized eigenvalues and eigenvectors of (A,B)
    BU = B @ U

    In:
        import numpy as np
        import scipy.linalg as sla
        N = 500
        A = np.random.randn(N,N)
        A = 0.5*(A + A.T)
        B = np.random.randn(N,N)
        B = sla.sqrtm(B.T @ B)
        dd_all, U_all = sla.eigh(A, B)
        inds = np.random.permutation(N)[:100]
        dd = dd_all[inds]
        U = U_all[:, inds]
        BU = B @ U
        sigma = np.random.randn()
        gamma = np.random.randn()
        b = np.random.randn(N)
        solve_A_minus_sigma_B = lambda z: np.linalg.solve(A - sigma*B, z)
        x = solve_shifted_deflated(b, solve_A_minus_sigma_B, sigma, gamma, dd, BU)
        x_true = np.linalg.solve(A - sigma * B + gamma * BU @ np.diag(dd) @ BU.T, b)
        err=np.linalg.norm(x - x_true) / np.linalg.norm(x_true)
        print('err=', err)

    Out:
        err= 9.937357654736434e-14
    '''
    N = b.shape[0]
    k = len(dd)
    assert(dd.shape == (k,))
    assert(BU.shape == (N,k))
    diag_Phi= (gamma * dd) * (dd - sigma) / ((dd - sigma) + (gamma * dd))
    return solve_A_minus_sigma_B(b - BU @ (diag_Phi * (BU.T @ solve_A_minus_sigma_B(b))))




@dataclass(frozen=True)
class DeflatedShiftedOperator:
    '''A - sigma*B + gamma * B @ U @ diag(dd) @ U.T @ B
    U.T @ A @ U = diag(dd)
    U.T @ B @ U = I
    solve_P(x) =approx= (A - sigma*B)^-1 @ x
    A and B are symmetric'''
    apply_A: vec2vec
    apply_B: vec2vec
    sigma: float
    solve_shifted_preconditioner: vec2vec # Approximates v -> (A - sigma*B)^-1 @ v
    gamma: float
    BU: np.ndarray # B @ U
    dd: np.ndarray

    @cached_property
    def N(me):
        return me.BU.shape[0]

    @cached_property
    def k(me):
        return len(me.dd)

    def __post_init__(me):
        assert(len(me.BU.shape) == 2)
        assert(len(me.dd.shape) == 1)
        assert(me.BU.shape == (me.N, me.k))
        assert(me.dd.shape == (me.k,))

        # b = np.random.randn(me.N)
        # x = me.solve_shifted(b)
        # norm_r = np.linalg.norm(b - me.apply_shifted(x))
        # norm_b = np.linalg.norm(b)
        # print('shifted: norm_r/norm_b=', norm_r / norm_b)
        # assert(norm_r < 1e-6 * norm_b)

        # b = np.random.randn(me.N)
        # x = me.solve_shifted_deflated(b)
        # norm_r = np.linalg.norm(b - me.apply_shifted_deflated(x))
        # norm_b = np.linalg.norm(b)
        # print('shifted and deflated: norm_r/norm_b=', norm_r / norm_b)
        # assert(norm_r < 1e-6 * norm_b)

    def apply_shifted(me, x: np.ndarray) -> np.ndarray: # x -> (A - sigma*B) @ x
        return me.apply_A(x) - me.sigma * me.apply_B(x)

    def solve_shifted(me, b: np.ndarray) -> np.ndarray: # b -> (A - sigma*B)^-1 @ b
        # return me.solve_P(b)
        return spla.gmres(spla.LinearOperator((me.N, me.N), matvec=me.apply_shifted), b,
                          M=spla.LinearOperator((me.N, me.N), matvec=me.solve_shifted_preconditioner),
                          tol=_GMRES_TOL)[0]

    def apply_deflated(me, x: np.ndarray): # x -> (A + gamma*B @ U @ diag(dd) @ U.T @ B) @ x
        return me.apply_A(x) + me.gamma * me.BU @ (me.dd * (me.BU.T @ x))

    # A - sigma * B + gamma * B @ U @ diag(dd) @ B @ U.T
    @cached_property
    def diag_Phi(me): # Phi = ((gamma*diag(dd)))^-1 - B @ U.T @ (A - sigma*B)^-1 @ B @ U)^-1
        return (me.gamma*me.dd) * (me.dd - me.sigma) / ((me.dd - me.sigma) + (me.gamma*me.dd))

    def apply_shifted_deflated(me, x: np.ndarray) -> np.ndarray: # x -> (A - sigma*B + gamma*B @ U @ diag(dd) @ U.T @ B) @ x
        return me.apply_shifted(x) + me.gamma * me.BU @ (me.dd * (me.BU.T @ x))

    def solve_shifted_deflated(me, b: np.ndarray) -> np.ndarray: # b -> (A - sigma*B + gamma*B @ U @ diag(dd) @ U.T @ B)^-1 @ b
        return solve_shifted_deflated(b, me.solve_shifted, me.sigma, me.gamma, me.dd, me.BU)
        # return me.solve_shifted(b - me.BU @ (me.diag_Phi * (me.BU.T @ me.solve_shifted(b))))

    def solve_shifted_preconditioner_deflated(me, b: np.ndarray) -> np.ndarray: # b -> (A - sigma*B + gamma*B @ U @ diag(dd) @ U.T @ B)^-1 @ b
        return me.solve_shifted_preconditioner(b - me.BU @ (me.diag_Phi * (me.BU.T @ me.solve_shifted_preconditioner(b))))

    def get_eigs_near_sigma(me, target_num_eigs=10, tol=1e-7, ncv_factor=None, maxiter=3, mode='cayley',
                            preconditioner_only=False) -> typ.Tuple[np.ndarray, np.ndarray]: # (dd, U)
        if ncv_factor is None:
            ncv = None
        else:
            ncv = ncv_factor*target_num_eigs

        if preconditioner_only:
            OPinv_matvec = me.solve_shifted_preconditioner_deflated
        else:
            OPinv_matvec = me.solve_shifted_deflated

        try:
            dd, U = spla.eigsh(spla.LinearOperator((me.N, me.N), matvec=me.apply_deflated), target_num_eigs,
                                  sigma=me.sigma,
                                  mode=mode,
                                  M=spla.LinearOperator((me.N, me.N), matvec=me.apply_B),
                                  OPinv=spla.LinearOperator((me.N, me.N), matvec=OPinv_matvec),
                                  which='LM', return_eigenvectors=True,
                                  tol=tol,
                                  ncv=ncv,
                                  maxiter=maxiter)
        except spla.ArpackNoConvergence as ex:
            U = ex.eigenvectors
            dd = ex.eigenvalues
            # print(ex)

        print(len(dd), ' / ', target_num_eigs, ' eigs found')

        return dd, U

    def update_sigma(me, new_sigma: float, new_solve_P: vec2vec) -> 'DeflatedShiftedOperator':
        return DeflatedShiftedOperator(me.apply_A, me.apply_B, new_sigma, new_solve_P, me.gamma, me.BU, me.dd)

    def update_gamma(me, new_gamma: float) -> 'DeflatedShiftedOperator':
        return DeflatedShiftedOperator(me.apply_A, me.apply_B, me.sigma, me.solve_shifted_preconditioner, new_gamma, me.BU, me.dd)

    def update_deflation(me, BU2: np.ndarray, dd2: np.ndarray) -> 'DeflatedShiftedOperator':
        new_BU = np.hstack([me.BU, BU2])
        new_dd = np.concatenate([me.dd, dd2])
        return DeflatedShiftedOperator(me.apply_A, me.apply_B, me.sigma, me.solve_shifted_preconditioner, me.gamma, new_BU, new_dd)

    # def update_deflation_with_reorthogonalization(me, U2: np.ndarray, dd2: np.ndarray) -> 'DeflatedShiftedOperator':
    #     BU2 = np.zeros(U2.shape)
    #     for k in range(BU2.shape[1]):
    #         BU2[:,k] = me.apply_B(U2[:,k])
    #     BU2_hat = BU2 - me.BU @ (me.BU.T @ U2)
    #     new_BU = np.hstack([me.BU, BU2_hat])
    #     new_dd = np.concatenate([me.dd, dd2])
    #     return DeflatedShiftedOperator(me.apply_A, me.apply_B, me.sigma, me.solve_shifted_preconditioner, me.gamma, new_BU, new_dd)


class CountedOperator:
    def __init__(me,
                 shape: typ.Tuple[int, int],
                 matvec: typ.Callable[[np.ndarray], np.ndarray],
                 display: bool = False,
                 name: str = '',
                 dtype=float
                 ):
        me.shape = shape
        me._matvec = matvec
        me.display = display
        me.count = 0
        me.name = name

    def matvec(me, x: np.ndarray) -> np.ndarray:
        assert(x.shape == (me.shape[1],))
        if me.display:
            print(me.name + ' count:', me.count)
        me.count += 1
        return me._matvec(x)

    def matmat(me, X: np.ndarray) -> np.ndarray:
        assert(len(X.shape) == 2)
        k = X.shape[1]
        assert(X.shape == (me.shape[1], k))
        Y = np.zeros((me.shape[0], k))
        for ii in range(k):
            Y[:,ii] = me.matvec(X[:,ii])
        return Y

    def __call__(me, x: np.ndarray) -> np.ndarray:
        if len(x.shape) == 1:
            return me.matvec(x)
        else:
            return me.matmat(x)

    def as_linear_operator(me):
        return spla.aslinearoperator(me)


def deflate_negative_eigs_near_sigma(DSO: DeflatedShiftedOperator,
                                     B_op: CountedOperator,
                                     threshold: float,
                                     chunk_size: int,
                                     ncv_factor: float,
                                     lanczos_maxiter: int,
                                     tol: float,
                                     preconditioner_only: bool=False,
                                     display: bool=True,
                                     max_tries=10,
                                     ) -> typ.Tuple[DeflatedShiftedOperator, float, float]: # DSO, band lower bound, band upper bound
    assert(threshold < 0.0)
    sigma = DSO.sigma
    dd = np.zeros(0)
    for _ in range(max_tries):
        dd_new, U_new = DSO.get_eigs_near_sigma(target_num_eigs=chunk_size, ncv_factor=ncv_factor,
                                                mode='cayley', maxiter=lanczos_maxiter, tol=tol,
                                                preconditioner_only=preconditioner_only)
        good_inds = (dd_new < threshold)
        if display:
            print('Updating deflation')
        if np.any(good_inds):
            dd = np.concatenate([dd, dd_new[good_inds]])
            DSO = DSO.update_deflation(
                B_op.matmat(U_new[:, good_inds]), dd_new[good_inds])
            # DSO = DSO.update_deflation_with_reorthogonalization(
            #     U_new[:, good_inds], dd_new[good_inds])

        if len(dd_new) == 0 or np.any(dd_new >= threshold):
            break

    if len(dd) > 0:
        d_lower = np.min(dd)
        d_upper = np.max(dd)
    else:
        d_lower = None
        d_upper = None

    return DSO, d_lower, d_upper


def check_symmetry(apply_A: vec2vec, N: int, rtol: float=1e-10) -> None:
    x = np.random.randn(N)
    Ax = apply_A(x)
    assert(len(Ax) == N)
    y = np.random.randn(N)
    Ay = apply_A(y)
    t1 = np.dot(y, Ax)
    t2 = np.dot(x, Ay)
    assert(np.abs(t1 - t2) <= rtol * (np.abs(t1) + np.abs(t2)))


def deflate_negative_eigenvalues(apply_A: vec2vec,
                                 apply_B: vec2vec,
                                 solve_B: vec2vec,
                                 N: int, # A.shape = B.shape = (N,N)
                                 make_OP: typ.Callable[[float], vec2vec], # Approximates sigma -> (v -> (A - sigma*B)^-1 @ v)
                                 threshold = -0.5,
                                 sigma_factor: float=7.0, # Sigma scaled up by this much above previous bound
                                 chunk_size=50,
                                 tol: float=1e-8,
                                 ncv_factor=3,
                                 lanczos_maxiter=2,
                                 display=True,
                                 perturb_mu_factor: float=1e-3,
                                 max_tries: int=100
                                ) -> typ.Tuple[np.ndarray, np.ndarray, float]: # (dd, V, LM_eig)
    '''Form low rank update A -> A + V @ diag(dd) @ V.T such that
        eigs(A + V @ diag(dd) @ V.T, B) > threshold
    A must be symmetric
    B must be symmetric positive definite
    OP = A - sigma*B
    OP_preconditioner = make_OP_preconditioner(sigma)
    OP_preconditioner(b) =approx= OP^-1 @ b

    In:
        import numpy as np
        import scipy.linalg as sla

        N = 1000
        A_diag = np.sort(np.random.randn(N))
        apply_A = lambda x: A_diag * x

        B_diag = np.random.randn(N)
        B_diag = np.sqrt(B_diag * B_diag)
        apply_B = lambda x: B_diag * x
        solve_B = lambda x: x / B_diag

        def make_shifted_solver(shift):
            OP_diag = A_diag - shift * B_diag
            return lambda x: x / OP_diag

        threshold = -0.5
        dd, V, LM_eig = deflate_negative_eigenvalues(apply_A, apply_B, solve_B, N,
                                                     make_shifted_solver,
                                                     threshold=threshold,
                                                     chunk_size=50,
                                                     display=True,
                                                    )

        A = np.diag(A_diag)
        B = np.diag(B_diag)
        ee_true, U_true = sla.eigh(A, B)

        A_deflated = A - V @ np.diag(dd) @ V.T

        ee_deflated_true = ee_true.copy()
        ee_deflated_true[ee_true < threshold] = 0.0
        A_deflated_true = (B @ U_true) @ np.diag(ee_deflated_true) @ (B @ U_true).T

        deflation_error = np.linalg.norm(A_deflated - A_deflated_true) / np.linalg.norm(A_deflated_true)
        print('deflation_error=', deflation_error)

    Out:
        Getting largest magnitude eigenvalue
        LM_eig= 3248.304697081123
        Getting eigenvalues in [-3280.787744051934, -0.5] via shift-and-invert method
        making A-sigma*B solver, sigma= -3.499931780024662
        Getting eigs near sigma= -3.499931780024662
        50  /  50  eigs found
        Updating deflation
        50  /  50  eigs found
        Updating deflation
        50  /  50  eigs found
        Updating deflation
        50  /  50  eigs found
        Updating deflation
        35  /  50  eigs found
        Updating deflation
        27  /  50  eigs found
        Updating deflation
        21  /  50  eigs found
        Updating deflation
        28  /  50  eigs found
        Updating deflation
        20  /  50  eigs found
        Updating deflation
        band_lower= -29.619909078864723
        making A-sigma*B preconditioner, sigma= -207.27956680733942
        Getting eigs near sigma= -207.27956680733942
        12  /  50  eigs found
        Updating deflation
        0  /  50  eigs found
        Updating deflation
        band_lower= -1450.9569676513759
        making A-sigma*B preconditioner, sigma= -3280.408481880921
        Getting eigs near sigma= -3280.408481880921
        1  /  50  eigs found
        Updating deflation
        band_lower= -22962.859373166448
        deflation_error= 1.0055708985814356e-08
    '''
    assert(N > 0)
    assert(threshold < 0.0)
    assert(tol > 0.0)
    assert(sigma_factor > 0.0)
    A_op = CountedOperator((N,N), apply_A, display=False, name='A')
    B_op = CountedOperator((N, N), apply_B, display=False, name='B')
    iB_op = CountedOperator((N, N), solve_B, display=False, name='invB')

    def printmaybe(*args, **kwargs):
        if display:
            print(*args, **kwargs)

    V0 = np.zeros((N, 0))
    dd0 = np.zeros((0,))

    # Get largest magnitude eigenvalue of matrix pencil (A,B)
    printmaybe('Getting largest magnitude eigenvalue')
    LM_eig = spla.eigsh(spla.LinearOperator((N,N), matvec=A_op.matvec),
                         1,
                         M=spla.LinearOperator((N,N), matvec=B_op.matvec),
                         Minv=spla.LinearOperator((N,N), matvec=iB_op.matvec),
                         which='LM', return_eigenvectors=False,
                         tol=tol)[0]
    printmaybe('LM_eig=', LM_eig)

    if -1.0 < -np.abs(LM_eig):
        return dd0, V0, LM_eig

    dd, V = negative_eigenvalues_of_matrix_pencil(
        apply_A, apply_B, make_OP, N, -np.abs(1.01*LM_eig), threshold, dd0, V0,
        sigma_factor=sigma_factor, chunk_size=chunk_size, tol=tol,
        ncv_factor=ncv_factor, lanczos_maxiter=lanczos_maxiter, display=display,
        perturb_mu_factor=perturb_mu_factor, max_tries=max_tries,
    )

    return dd, V, LM_eig


def negative_eigenvalues_of_matrix_pencil(
        apply_A: vec2vec,
        apply_B: vec2vec,
        make_OP: typ.Callable[[float], vec2vec], # Approximates sigma -> (v -> (A - sigma*B)^-1 @ v)
        N: int, # A.shape = B.shape = (N,N)
        range_min: float,
        range_max: float,
        solve_B: vec2vec=None, # Must be supplied if range_min is None
        prior_dd: np.ndarray=None,
        prior_V: np.ndarray=None,
        deflation_gamma: float=-2.0,
        sigma_factor: float=5.0, # Sigma scaled up by this much above previous bound
        chunk_size=50,
        tol: float=1e-8,
        ncv_factor=3,
        lanczos_maxiter=2,
        display=False,
        perturb_mu_factor: float=1e-3,
        max_tries=20,
) -> typ.Tuple[np.ndarray, np.ndarray]: # (eigs, evecs)
    '''Get generalized eigenvalues of (A,B) in (range_min, range_max) < 0.
    Generalized eigenvalues of (A,B) may cluster at zero or positive numbers, but must not cluster at negative numbers

    A must be symmetric
    B must be symmetric positive definite
    OP = A - sigma*B
    OP_preconditioner = make_OP_preconditioner(sigma)
    OP_preconditioner(b) =approx= OP^-1 @ b

    In:
        import numpy as np
        import scipy.linalg as sla

        N = 2000
        A_diag = np.sort(np.random.randn(N))
        apply_A = lambda x: A_diag * x

        B_diag = np.random.randn(N)
        B_diag = np.sqrt(B_diag * B_diag)
        apply_B = lambda x: B_diag * x
        solve_B = lambda x: x / B_diag

        def make_shifted_solver(shift):
            OP_diag = A_diag - shift * B_diag
            return lambda x: x / OP_diag

        range1_max = -0.1
        range1_min = -50.0
        range2_max = -100.0
        range2_min = -300.0

        dd1, V1 = get_negative_eigenvalues_in_range(
            apply_A, apply_B, make_shifted_solver, N, range1_min, range1_max, display=True)

        dd2, V2 = get_negative_eigenvalues_in_range(
            apply_A, apply_B, make_shifted_solver, N, range2_min, range2_max,
            prior_dd=dd1, prior_V=V1, display=True)

        A = np.diag(A_diag)
        B = np.diag(B_diag)

        A_deflated = A - V1 @ np.diag(dd1) @ V1.T - V2 @ np.diag(dd2) @ V2.T

        ee_true, U_true = sla.eigh(A, B)

        zeroing_inds1 = np.logical_and(range1_min < ee_true, ee_true < range1_max)
        zeroing_inds2 = np.logical_and(range2_min < ee_true, ee_true < range2_max)
        zeroing_inds12 = np.logical_or(zeroing_inds1, zeroing_inds2)

        ee_deflated_true = ee_true.copy()
        ee_deflated_true[zeroing_inds12] = 0.0
        A_deflated_true = (B @ U_true) @ np.diag(ee_deflated_true) @ (B @ U_true).T

        deflation_error = np.linalg.norm(A_deflated_true - A_deflated) / np.linalg.norm(A_deflated_true)
        print('deflation_error=', deflation_error)

    Out:
        Getting eigenvalues in [-50.0, -0.1] via shift-and-invert method
        making A-sigma*B solver, sigma= -0.70027346246369
        Getting eigs near sigma= -0.70027346246369
        50  /  50  eigs found
        Updating deflation
        50  /  50  eigs found
        Updating deflation
        50  /  50  eigs found
        Updating deflation
        50  /  50  eigs found
        Updating deflation
        50  /  50  eigs found
        Updating deflation
        47  /  50  eigs found
        Updating deflation
        50  /  50  eigs found
        Updating deflation
        49  /  50  eigs found
        Updating deflation
        46  /  50  eigs found
        Updating deflation
        50  /  50  eigs found
        Updating deflation
        43  /  50  eigs found
        Updating deflation
        35  /  50  eigs found
        Updating deflation
        29  /  50  eigs found
        Updating deflation
        44  /  50  eigs found
        Updating deflation
        38  /  50  eigs found
        Updating deflation
        22  /  50  eigs found
        Updating deflation
        22  /  50  eigs found
        Updating deflation
        29  /  50  eigs found
        Updating deflation
        36  /  50  eigs found
        Updating deflation
        band_lower= -4.901914237245831
        making A-sigma*B preconditioner, sigma= -34.31145635256429
        Getting eigs near sigma= -34.31145635256429
        50  /  50  eigs found
        Updating deflation
        50  /  50  eigs found
        Updating deflation
        33  /  50  eigs found
        Updating deflation
        0  /  50  eigs found
        Updating deflation
        band_lower= -372.83655294440257
        Getting eigenvalues in [-300.0, -100.0] via shift-and-invert method
        making A-sigma*B solver, sigma= -300.09893022873047
        Getting eigs near sigma= -300.09893022873047
        19  /  50  eigs found
        Updating deflation
        band_lower= -2100.6925116011134
        deflation_error= 7.59970968906748e-10
    '''
    assert(N > 0)
    assert(range_max < 0.0)
    assert(tol > 0.0)
    assert(sigma_factor > 0.0)
    B_op = CountedOperator((N, N), apply_B, display=False, name='B')

    def printmaybe(*args, **kwargs):
        if display:
            print(*args, **kwargs)

    if (prior_dd is None) or (prior_V is None):
        assert(prior_dd is None)
        assert(prior_V is None)
        prior_dd = np.zeros((0,))
        prior_V = np.zeros((N,0))

    num_prior_dd = len(prior_dd)
    assert(prior_dd.shape == (num_prior_dd,))
    assert(prior_V.shape == (N, num_prior_dd))

    if range_min is None:
        if solve_B is None:
            raise RuntimeError('solve_B must be supplied if range_min is not supplied')
        printmaybe('range_min=None => range_min=-|LM_eig|')
        printmaybe('getting largest magnitude eigenvalue, LM_eig')
        LM_eig = spla.eigsh(spla.LinearOperator((N,N), matvec=apply_A),
                             1,
                             M=spla.LinearOperator((N,N), matvec=apply_B),
                             Minv=spla.LinearOperator((N,N), matvec=solve_B),
                             which='LM', return_eigenvectors=False,
                             tol=tol)[0]
        printmaybe('LM_eig=', LM_eig)
        range_min = -np.abs(LM_eig)

    printmaybe('Getting eigenvalues in [' + str(range_min) + ', ' + str(range_max) + '] via shift-and-invert method')

    def perturb(x):
        return (1.0 + perturb_mu_factor * (np.random.rand() - 0.5)) * x

    dd = prior_dd.copy()
    V = prior_V.copy()

    band_lower = range_max
    while range_min <= band_lower:
        proposed_sigma = band_lower * sigma_factor
        sigma = perturb(np.max([range_min, proposed_sigma]))

        printmaybe('making A-sigma*B preconditioner, sigma=', sigma)
        solve_P = make_OP(sigma)
        iP_op = CountedOperator((N,N), solve_P, display=False, name='invP')
        DSO = DeflatedShiftedOperator(
            apply_A, apply_B, sigma, iP_op.matvec, deflation_gamma, # -2.0: flip eigs across zero to get them out of the way
            V, dd)

        printmaybe('Getting eigs near sigma=', sigma)
        DSO, d_lower, _ = deflate_negative_eigs_near_sigma(
            DSO, B_op,
            range_max, #band_lower,
            chunk_size,
            ncv_factor, lanczos_maxiter, tol,
            max_tries=max_tries, preconditioner_only=True, display=display)

        dd = DSO.dd
        V = DSO.BU

        if d_lower is None:
            band_lower = sigma * sigma_factor
        else:
            band_lower = np.min([sigma * sigma_factor, d_lower])
        printmaybe('band_lower=', band_lower)

    new_V = V[:,num_prior_dd:].reshape((N,-1))
    new_dd = dd[num_prior_dd:].reshape(-1)

    new_good_inds = np.logical_and(range_min <= new_dd, new_dd < range_max)

    return new_dd[new_good_inds].reshape(-1), new_V[:, new_good_inds].reshape((N,-1))



