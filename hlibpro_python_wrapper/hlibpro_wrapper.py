import numpy as np
import scipy.sparse.linalg as spla
from scipy.optimize import root_scalar
from scipy.io import savemat
from time import time
import typing as typ
from dataclasses import dataclass
from functools import cached_property
from . import hlibpro_bindings as hpro_cpp
from .deflate_negative_eigenvalues import DeflatedShiftedOperator, negative_eigenvalues_of_matrix_pencil, solve_shifted_deflated
from .interpolate_shifted_inverses import shifted_inverse_interpolation_preconditioner

_DEFAULT_RTOL = 1e-7
_DEFAULT_ATOL = 1e-12

_MU_SPACING_FACTOR: float = 5.0e1
_BOUNDARY_MU_RTOL: float = 0.1
_MAX_MUS = 20


class HMatrix:
    def __init__(me, hmatrix_cpp_object, bct):
        # Don't touch underscore prefix variables. Access via @property methods if needed
        me._cpp_object = hmatrix_cpp_object
        me._bct = bct
        me._row_ct = me._bct.row_ct
        me._col_ct = me._bct.col_ct

        me.shape = (me.cpp_object.rows(), me._cpp_object.cols())
        me.dtype = np.double # Real: Complex not supported currently

    @property
    def cpp_object(me):
        return me._cpp_object

    @property
    def bct(me):
        return me._bct

    @property
    def row_ct(me):
        return me._row_ct

    @property
    def col_ct(me):
        return me._col_ct

    def byte_size(me):
        return me.cpp_object.byte_size()

    def copy(me):
        return HMatrix(hpro_cpp.copy_TMatrix(me.cpp_object), me.bct)

    def copy_struct(me):
        return HMatrix(hpro_cpp.copy_struct_TMatrix(me.cpp_object), me.bct)

    def copy_to(me, other):
        hpro_cpp.copy_TMatrix_into_another_TMatrix(me.cpp_object, other.cpp_object)
        other._bct = me.bct

    def transpose(me, overwrite=False):
        if overwrite:
            me._cpp_object.transpose()
            return me
        else:
            transposed_cpp_object = hpro_cpp.copy_TMatrix(me.cpp_object)
            transposed_cpp_object.transpose()
            return HMatrix(transposed_cpp_object, me.bct)

    @property
    def T(me):
        return me.transpose()

    def sym(me, rtol=_DEFAULT_RTOL, atol=_DEFAULT_ATOL, overwrite=False):
        if overwrite:
            A_sym = h_add(me.T, me, alpha=0.5, beta=0.5, rtol=rtol, atol=atol, overwrite_B=True)
        else:
            A_sym = h_add(me, me.T, alpha=0.5, beta=0.5, rtol=rtol, atol=atol, overwrite_B=True)
        # A_sym._set_symmetric()
        return A_sym

    def spd(me, **kwargs):
        return make_hmatrix_spd_hackbusch_kress_2007(me, **kwargs)
        # return rational_positive_definite_approximation_low_rank_method(me, **kwargs)
        # return rational_positive_definite_approximation_method1(me, **kwargs)

    def _set_symmetric(me):
        me._cpp_object.set_symmetric()

    def _set_nonsym(me):
        me._cpp_object.set_nonsym()

    def matvec(me, x):
        return h_matvec(me, x)

    def rmatvec(me, x):
        return h_rmatvec(me, x)

    def __add__(me, other):
        if isinstance(other, HMatrix):
            return h_add(other, me)

        else:
            raise RuntimeError('cannot add HMatrix to ' + str(other.type))

    def __iadd__(me, other):
        if isinstance(other, HMatrix):
            return h_add(other, me, overwrite_B=True)
        else:
            raise RuntimeError('cannot add HMatrix to ' + str(other.type))

    def __sub__(me, other, rtol=_DEFAULT_RTOL, atol=_DEFAULT_ATOL):
        if isinstance(other, HMatrix):
            return h_add(other, me, beta=-1.0)

        else:
            raise RuntimeError('cannot add HMatrix to ' + str(other.type))

    def __isub__(me, other):
        if isinstance(other, HMatrix):
            return h_add(other, me, beta=-1.0, overwrite_B=True)

        else:
            raise RuntimeError('cannot add HMatrix to ' + str(other.type))

    def __neg__(me):
        return h_scale(me, -1.0)

    def __mul__(me, other):
        if isinstance(other, HMatrix):
            return h_mul(me, other)

        if isinstance(other, float) or isinstance(other, np.number):
            return h_scale(me, other)

        if isinstance(other, np.ndarray) and ( other.shape == (me.shape[1],) ):
            return me.matvec(other)

        else:
            raise RuntimeError('cannot multiply HMatrix with ' + str(other.type))

    # def __imul__(me, other):
    #     if isinstance(other, HMatrix):
    #         return h_mul(me, other, alpha_A_B_hmatrix=me)
    #
    #     if isinstance(other, float) or isinstance(other, np.number):
    #         return h_scale(me, other, overwrite_A=True)
    #
    #     else:
    #         raise RuntimeError('cannot multiply HMatrix with ' + str(other.type))

    def __rmul__(me, other):
        if isinstance(other, HMatrix):
            return h_mul(other, me)

        if isinstance(other, float) or isinstance(other, np.number):
            return h_scale(me, other)

        if isinstance(other, np.ndarray) and ( other.shape == (me.shape[0],) ):
            return me.rmatvec(other)

        else:
            raise RuntimeError('cannot right multiply HMatrix with ' + str(other.type))

    def __matmul__(me, other):
        return me.__mul__(me, other)

    def __rmatmul__(me, other):
        return me.__rmul__(me, other)

    # def __imatmul__(me, other):
    #     return me.__imul__(me, other)

    def inv(me, rtol=_DEFAULT_RTOL, atol=_DEFAULT_ATOL,
            overwrite=False, display_progress=False,
            diag_type='general_diag', storage_type='store_normal', do_coarsen=False):
        return h_inv(me, rtol=rtol, atol=atol,
                     overwrite=overwrite, display_progress=display_progress,
                     diag_type=diag_type, storage_type=storage_type, do_coarsen=do_coarsen)

    def factorized_inverse(me, rtol=_DEFAULT_RTOL, atol=_DEFAULT_ATOL, overwrite=False):
        return h_factorized_inverse(me, rtol=rtol, atol=atol)

    def add_identity(me, s=1.0, overwrite=False):
        return add_identity_to_hmatrix(me, s=s, overwrite=overwrite)

    def mul_diag_left(me, v):
        hpro_cpp.mul_diag_left_wrapper(v, me.cpp_object, me.row_ct.cpp_object)

    def mul_diag_right(me, v):
        hpro_cpp.mul_diag_right_wrapper(v, me.cpp_object, me.col_ct.cpp_object)

    def visualize(me, filename):
        hpro_cpp.visualize_hmatrix(me.cpp_object, filename)

    def as_linear_operator(me):
        return spla.LinearOperator(me.shape, matvec=me.matvec, rmatvec=me.rmatvec)

    def low_rank_update(me, X, Y, overwrite=False, rtol=_DEFAULT_RTOL, atol=_DEFAULT_ATOL): #A -> A + X*Y
        X2 = np.zeros(X.shape)
        X2[:] = X
        Y2T = np.zeros(Y.shape[::-1])
        Y2T[:] = Y.T
        XY = hpro_cpp.make_permuted_hlibpro_low_rank_matrix(X2, Y2T, me.row_ct.cpp_object, me.col_ct.cpp_object)

        acc = hpro_cpp.TTruncAcc(relative_eps=rtol, absolute_eps=atol)

        if overwrite == False:
            A_plus_XY = me.copy()
        else:
            A_plus_XY = me

        hpro_cpp.add(1.0, XY, 1.0, A_plus_XY.cpp_object, acc)
        return A_plus_XY

    def dfp_update(me, X, Y,
                   overwrite=False,
                   force_positive_definite=True,
                   check_correctness=True,
                   rtol=_DEFAULT_RTOL,
                   atol=_DEFAULT_ATOL):
        # Y = A * X
        # A1 = (I - Y (Y^T X)^-1 X^T) A0 (I - X (X^T Y)^-1 Y^T) + Y (Y^T X)^-1 Y^T
        #    = (I - Q X^T) A0 (I - X Q^T) + Q Y^T,                   where Q := Y (Y^T X)^-1
        #    = A0 - A0 X Q^T - Q X^T A0 + Q X^T A0 X Q^T + Q Y^T
        #    = A0 - Y0 Q^T    - Q Y0^T    + Q X^T Y0 Q^T    + Q Y^T,    where Y0 := A0 X
        #    = A0 + (Q X^T Y0 - Y0) Q^T + Q (Y - Y0)^T
        if len(X.shape) == 1:
            X = X.reshape((-1,1))
        if len(Y.shape) == 1:
            Y = Y.reshape((-1,1))

        # A X = Y
        # Y^T X = X^T A X

        YtX = np.dot(Y.T, X)
        # print('DFP: YtX=', YtX)
        ee, P = np.linalg.eigh(YtX) # note: symmetric because Y = A*X
        # print('dfp_update: YtX eigs=', ee)
        positive_inds = (ee > 0)
        n_plus = np.sum(positive_inds)
        n = len(ee)
        n_neg = n - n_plus
        if n_neg > 0:
            print('warning: ', n_neg, ' / ', n, ' negative directions detected in DFP update')

        N = X.shape[0]
        P_plus = P[:, positive_inds].reshape((n, n_plus))
        XP_plus = np.dot(X, P_plus).reshape((N, n_plus))
        X_plus, RR = np.linalg.qr(XP_plus, mode='reduced')
        X_plus = X_plus.reshape((N, n_plus))
        RR = RR.reshape((n_plus, n_plus))
        C_plus = np.linalg.solve(RR.T, P_plus.T).T.reshape((n, n_plus))

        if check_correctness:
            err_X_plus = np.linalg.norm(np.dot(X, C_plus) - X_plus) / np.linalg.norm(X_plus)
            print('err_X_plus=', err_X_plus)

        # X_plus = np.dot(X, P[:, positive_inds])
        # C_plus = np.linalg.lstsq(X, X_plus)
        Y_plus = np.dot(Y, C_plus)

        YtX_plus = np.dot(Y_plus.T, X_plus)
        if check_correctness:
            ee_plus, _ = np.linalg.eigh(YtX_plus)
            num_negative_eigs_that_should_be_positive = np.sum(ee_plus < 0)
            print('num_negative_eigs_that_should_be_positive=', num_negative_eigs_that_should_be_positive, ' / ', n_plus )

        Q_plus = np.linalg.solve(YtX_plus, Y_plus.T).T

        Y0_plus = np.zeros((N, n_plus))
        for k in range(n_plus):
            Y0_plus[:,k] = me.matvec(X_plus[:,k].copy())

        L_plus = np.hstack([np.dot(Q_plus, np.dot(X_plus.T, Y0_plus)) - Y0_plus, Q_plus])
        R_plus = np.hstack([Q_plus, Y_plus - Y0_plus]).T

        print('doing low rank update')
        A1 = me.low_rank_update(L_plus, R_plus, overwrite=overwrite, rtol=rtol, atol=atol)

        if check_correctness: # Does Y = A*X after update?
            Y2_plus = np.zeros(Y_plus.shape)
            for k in range(X_plus.shape[1]):
                Y2_plus[:,k] = A1.matvec(X_plus[:,k])
            err_dfp = np.linalg.norm(Y2_plus - Y_plus) / np.linalg.norm(Y_plus)
            print('err_dfp=', err_dfp, ', rtol=', rtol, ', atol=', atol)

        # if force_positive_definite:
        #     iee = np.zeros(len(ee))
        #     iee[ee > 0] = 1./ee[ee > 0]
        #     # iee = 1./np.abs(ee)
        # else:
        #     iee = 1./ee
        # iYtX = np.dot(P, np.dot(np.diag(iee), P.T))
        # Q = np.dot(Y, iYtX)
        #
        # Y0 = np.zeros((me.shape[0], X.shape[1]))
        # for k in range(X.shape[1]):
        #     Y0[:,k] = me.matvec(X[:,k].copy())
        #
        # L = np.hstack([np.dot(Q, np.dot(X.T, Y0)) - Y0, Q])
        # R = np.hstack([Q, Y - Y0]).T
        #
        # print('doing low rank update')
        # A1 = me.low_rank_update(L, R, overwrite=overwrite, rtol=rtol, atol=atol)
        #
        # if check_correctness: # Does Y = A*X after update?
        #     Y2 = np.zeros(Y.shape)
        #     for k in range(X.shape[1]):
        #         Y2[:,k] = A1.matvec(X[:,k])
        #     err_dfp = np.linalg.norm(Y2 - Y) / np.linalg.norm(Y)
        #     print('err_dfp=', err_dfp, ', rtol=', rtol, ', atol=', atol)

        return A1

    def broyden_update(me, X, Y,
                       overwrite=False,
                       check_correctness=True,
                       rtol=_DEFAULT_RTOL,
                       atol=_DEFAULT_ATOL):
        # A1 = A0 + (Y - A0 X)(X^T X)^-1 X^T
        if len(X.shape) == 1:
            X = X.reshape((-1, 1))
        if len(Y.shape) == 1:
            Y = Y.reshape((-1, 1))

        Y0 = np.zeros((me.shape[0], X.shape[1]))
        for k in range(X.shape[1]):
            Y0[:,k] = me.matvec(X[:,k].copy())

        L = Y - Y0
        R = np.linalg.solve(np.dot(X.T, X), X.T)

        print('doing low rank update')
        A1 = me.low_rank_update(L, R, overwrite=overwrite, rtol=rtol, atol=atol)

        if check_correctness:
            Y2 = np.zeros(Y.shape)
            for k in range(X.shape[1]):
                Y2[:,k] = A1.matvec(X[:,k])
            err_broyden = np.linalg.norm(Y2 - Y) / np.linalg.norm(Y)
            print('err_broyden=', err_broyden, ', rtol=', rtol, ', atol=', atol)

        return A1

    def SRK_update(me, X, Y,
                   overwrite=False,
                   check_correctness=True,
                   rtol=_DEFAULT_RTOL,
                   atol=_DEFAULT_ATOL):
        # A1 = A0 + (Y - A0 X)((Y - A0 X)^T X)^-1 (Y - A0 X)^T
        if len(X.shape) == 1:
            X = X.reshape((-1, 1))
        if len(Y.shape) == 1:
            Y = Y.reshape((-1, 1))

        Y0 = np.zeros((me.shape[0], X.shape[1]))
        for k in range(X.shape[1]):
            Y0[:,k] = me.matvec(X[:,k].copy())

        L = Y - Y0
        R = np.linalg.solve(np.dot(L.T, X), L.T)

        print('doing low rank update')
        A1 = me.low_rank_update(L, R, overwrite=overwrite, rtol=rtol, atol=atol)

        if check_correctness:
            Y2 = np.zeros(Y.shape)
            for k in range(X.shape[1]):
                Y2[:,k] = A1.matvec(X[:,k])
            err_SRK = np.linalg.norm(Y2 - Y) / np.linalg.norm(Y)
            print('err_SRK=', err_SRK, ', rtol=', rtol, ', atol=', atol)

        return A1


def add_identity_to_hmatrix(A_hmatrix, s=1.0, overwrite=False):
    if overwrite:
        A_plus_alpha_I = A_hmatrix
    else:
        A_plus_alpha_I = A_hmatrix.copy()
    hpro_cpp.add_identity_to_hmatrix(A_plus_alpha_I.cpp_object, s)
    return A_plus_alpha_I


def h_add(A_hmatrix, B_hmatrix, alpha=1.0, beta=1.0, rtol=_DEFAULT_RTOL, atol=_DEFAULT_ATOL, overwrite_B=False):
    # C = A + alpha * B to tolerance given by truncation accuracy object acc
    acc = hpro_cpp.TTruncAcc(relative_eps=rtol, absolute_eps=atol)
    if overwrite_B:
        alpha_A_plus_beta_B_hmatrix = B_hmatrix
    else:
        alpha_A_plus_beta_B_hmatrix = B_hmatrix.copy()
    hpro_cpp.add(alpha, A_hmatrix.cpp_object, beta, alpha_A_plus_beta_B_hmatrix.cpp_object, acc)
    return alpha_A_plus_beta_B_hmatrix


def h_scale(A_hmatrix, alpha, overwrite_A=False):
    # C = alpha * A
    if overwrite_A:
        alpha_A_hmatrix = A_hmatrix
    else:
        alpha_A_hmatrix = A_hmatrix.copy()
    alpha_A_hmatrix.cpp_object.scale(alpha)
    return alpha_A_hmatrix


def h_mul(A_hmatrix, B_hmatrix, alpha_A_B_hmatrix=None, alpha=1.0, rtol=_DEFAULT_RTOL, atol=_DEFAULT_ATOL, display_progress=True):
    # C = A * B
    acc = hpro_cpp.TTruncAcc(relative_eps=rtol, absolute_eps=atol)
    if alpha_A_B_hmatrix is None:
        # AB_admissibility_eta = np.min([A_hmatrix.admissibility_eta, B_hmatrix.admissibility_eta])
        # AB_bct = build_block_cluster_tree(A_hmatrix.row_ct, B_hmatrix.col_ct, AB_admissibility_eta)
        alpha_A_B_hmatrix = A_hmatrix.copy_struct()

    if display_progress:
        hpro_cpp.multiply_with_progress_bar(alpha, hpro_cpp.apply_normal, A_hmatrix.cpp_object,
                                            hpro_cpp.apply_normal, B_hmatrix.cpp_object,
                                            0.0, alpha_A_B_hmatrix.cpp_object, acc)
    else:
        hpro_cpp.multiply_without_progress_bar(alpha, hpro_cpp.apply_normal, A_hmatrix.cpp_object,
                                               hpro_cpp.apply_normal, B_hmatrix.cpp_object,
                                               0.0, alpha_A_B_hmatrix.cpp_object, acc)
    return alpha_A_B_hmatrix


class FactorizedInverseHMatrix:
    def __init__(me, factorized_inverse_cpp_object, factors_cpp_object, inverse_bct):
        me._cpp_object = factorized_inverse_cpp_object
        me._factors_cpp_object = factors_cpp_object
        me._bct = inverse_bct
        me._row_ct = me.bct.row_ct
        me._col_ct = me.bct.col_ct

        me.shape = (me._factors_cpp_object.rows(), me._factors_cpp_object.cols())
        me.dtype = np.double # Real: Complex not supported currently

    @property
    def cpp_object(me):
        return me._cpp_object

    @property
    def factors_cpp_object(me):
        return me._factors_cpp_object

    @property
    def bct(me):
        return me._bct

    @property
    def row_ct(me):
        return me._row_ct

    @property
    def col_ct(me):
        return me._col_ct

    def matvec(me, x):
        return h_factorized_solve(me, x)


def h_factorized_inverse(A_hmatrix, rtol=_DEFAULT_RTOL, atol=_DEFAULT_ATOL, overwrite=False):
    acc = hpro_cpp.TTruncAcc(relative_eps=rtol, absolute_eps=atol)
    if overwrite:
        factors_cpp_object = A_hmatrix.cpp_object
    else:
        factors_cpp_object = hpro_cpp.copy_TMatrix(A_hmatrix.cpp_object)
    cpp_object = hpro_cpp.factorize_inv_with_progress_bar(factors_cpp_object, acc)
    return FactorizedInverseHMatrix(cpp_object, factors_cpp_object, A_hmatrix.bct)


class FactorizedHMatrix:
    def __init__(me, eval_cpp_object, eval_inverse_cpp_object, factors_cpp_object, bct,
                 eval_type, storage_type, coarsened, factorization_type):
        me._eval_cpp_object = eval_cpp_object
        me._eval_inverse_cpp_object = eval_inverse_cpp_object
        me._factors_cpp_object = factors_cpp_object
        me._eval_type = eval_type
        me._storage_type = storage_type
        me._coarsened = coarsened
        me._factorization_type = factorization_type
        me._bct = bct
        me._row_ct = me.bct.row_ct
        me._col_ct = me.bct.col_ct

        me.shape = (me._factors_cpp_object.rows(), me._factors_cpp_object.cols())
        me.dtype = np.double # Real: Complex not supported currently

    @property
    def eval_cpp_object(me):
        return me._eval_cpp_object

    @property
    def eval_inverse_cpp_object(me):
        return me._eval_inverse_cpp_object

    @property
    def factors_cpp_object(me):
        return me._factors_cpp_object

    @property
    def eval_type(me):
        return me._eval_type

    @property
    def storage_type(me):
        return me._storage_type

    @property
    def coarsened(me):
        return me._coarsened

    @property
    def factorization_type(me):
        return me._factorization_type

    @property
    def bct(me):
        return me._bct

    @property
    def row_ct(me):
        return me._row_ct

    @property
    def col_ct(me):
        return me._col_ct

    def solve(me, y):
        return hpro_cpp.h_factorized_inverse_matvec(me.eval_inverse_cpp_object,
                                                    me.row_ct.cpp_object,
                                                    me.col_ct.cpp_object, y)

    def apply(me, x):
        return hpro_cpp.h_factorized_matvec(me.eval_cpp_object,
                                            me.row_ct.cpp_object,
                                            me.col_ct.cpp_object, x)

    def as_linear_operator(me, inverse=False):
        if inverse:
            return spla.LinearOperator(me.shape, matvec=me.solve)
        else:
            return spla.LinearOperator(me.shape, matvec=me.apply)

    def visualize(me, filename):
        hpro_cpp.visualize_hmatrix(me.factors_cpp_object, filename)

    def split(me):
        if me.eval_type == 'point_wise':
            cpp_eval_type = hpro_cpp.eval_type_t.point_wise
        elif me.eval_type == 'block_wise':
            cpp_eval_type = hpro_cpp.eval_type_t.block_wise
        else:
            raise RuntimeError('eval_type must be point_wise or block_wise. eval_type=', me.eval_type)

        if me.storage_type == 'store_normal':
            cpp_storage_type = hpro_cpp.storage_type_t.store_normal
        elif me.storage_type == 'store_inverse':
            cpp_storage_type = hpro_cpp.storage_type_t.store_inverse
        else:
            raise RuntimeError('storage_type must be store_normal or store_inverse. storage_type=', me.storage_type)

        fac_options = hpro_cpp.fac_options_t(cpp_eval_type, cpp_storage_type, me.coarsened)

        if me.factorization_type == 'LDL':
            LD_list = hpro_cpp.split_ldl_factorization(me._factors_cpp_object, fac_options)

            L = HMatrix(LD_list[0], me.bct)
            D = HMatrix(LD_list[1], me.bct)
            return L, D
        elif me.factorization_type == 'LU':
            LU_list = hpro_cpp.split_lu_factorization(me._factors_cpp_object, fac_options)

            L = HMatrix(LU_list[0], me.bct)
            U = HMatrix(LU_list[1], me.bct)
            return L, U
        else:
            raise RuntimeError('asdf')

def h_inv(A_hmatrix, rtol=_DEFAULT_RTOL, atol=_DEFAULT_ATOL,
          overwrite=False, display_progress=False,
          diag_type='general_diag', storage_type='store_normal', do_coarsen=False):
    # Look in hpro/algebra/mat_inv.hh
    acc = hpro_cpp.TTruncAcc(relative_eps=rtol, absolute_eps=atol)

    if do_coarsen:
        raise RuntimeError('coarsening inverse of hmatrix not implemented yet in python wrapper')

    if overwrite:
        inverse_cpp_object = A_hmatrix.cpp_object
    else:
        inverse_cpp_object = hpro_cpp.copy_TMatrix(A_hmatrix.cpp_object)

    if diag_type == 'general_diag':
        cpp_diag_type = hpro_cpp.diag_type_t.general_diag
    elif diag_type == 'unit_diag':
        cpp_diag_type = hpro_cpp.diag_type_t.unit_diag
    else:
        raise RuntimeError('diag_type must be general_diag or unit_diag. diag_type=', diag_type)

    if storage_type == 'store_normal':
        cpp_storage_type = hpro_cpp.storage_type_t.store_normal
    elif storage_type == 'store_inverse':
        cpp_storage_type = hpro_cpp.storage_type_t.store_inverse
    else:
        raise RuntimeError('storage_type must be store_normal or store_inverse. storage_type=', storage_type)

    if display_progress:
        progress_bar = hpro_cpp.TConsoleProgressBar()
        inv_options = hpro_cpp.inv_options_t(cpp_diag_type, cpp_storage_type, do_coarsen, progress_bar)
    else:
        # inv_options = hpro_cpp.fac_options_t(cpp_diag_type, cpp_storage_type, do_coarsen)
        # inv_options = hpro_cpp.fac_options_t(cpp_diag_type, cpp_storage_type, do_coarsen, None)
        # inv_options = hpro_cpp.fac_options_t()
        inv_options = hpro_cpp.inv_options_t(cpp_diag_type, cpp_storage_type, do_coarsen, None)

    print('━━ H-matrix inverse ( rtol = ', rtol, ', atol = ', atol, ', overwrite=', overwrite, ' )')

    t = time()
    hpro_cpp.invert_h_matrix(inverse_cpp_object, acc, inv_options)
    dt_inv = time() - t

    print('    done in ', dt_inv)
    print('    size of inverse = ', inverse_cpp_object.byte_size(), ' bytes')

    if overwrite:
        return A_hmatrix
    else:
        return HMatrix(inverse_cpp_object, A_hmatrix.bct)


def h_ldl(A_hmatrix, rtol=_DEFAULT_RTOL, atol=_DEFAULT_ATOL,
          overwrite=False, display_progress=True,
          eval_type='block_wise', storage_type='store_normal', do_coarsen=False):
    return _factorize_h_matrix(A_hmatrix, 'LDL', rtol, atol,
                               overwrite, display_progress,
                               eval_type, storage_type, do_coarsen)


def h_lu(A_hmatrix, rtol=_DEFAULT_RTOL, atol=_DEFAULT_ATOL,
         overwrite=False, display_progress=True,
         eval_type='block_wise', storage_type='store_normal', do_coarsen=False):
    return _factorize_h_matrix(A_hmatrix, 'LU', rtol, atol,
                               overwrite, display_progress,
                               eval_type, storage_type, do_coarsen)


def _factorize_h_matrix(A_hmatrix, operation_to_perform, rtol, atol,
                        overwrite, display_progress,
                        eval_type, storage_type, do_coarsen):
    # Look in hpro/algebra/mat_fac.hh
    acc = hpro_cpp.TTruncAcc(relative_eps=rtol, absolute_eps=atol)

    if overwrite:
        factors_cpp_object = A_hmatrix.cpp_object
    else:
        factors_cpp_object = hpro_cpp.copy_TMatrix(A_hmatrix.cpp_object)

    if eval_type == 'point_wise':
        cpp_eval_type = hpro_cpp.eval_type_t.point_wise
    elif eval_type == 'block_wise':
        cpp_eval_type = hpro_cpp.eval_type_t.block_wise
    else:
        raise RuntimeError('eval_type must be point_wise or block_wise. eval_type=', eval_type)

    if storage_type == 'store_normal':
        cpp_storage_type = hpro_cpp.storage_type_t.store_normal
    elif storage_type == 'store_inverse':
        cpp_storage_type = hpro_cpp.storage_type_t.store_inverse
    else:
        raise RuntimeError('storage_type must be store_normal or store_inverse. storage_type=', storage_type)

    if display_progress:
        progress_bar = hpro_cpp.TConsoleProgressBar()
        fac_options = hpro_cpp.fac_options_t(cpp_eval_type, cpp_storage_type, do_coarsen, progress_bar)
    else:
        fac_options = hpro_cpp.fac_options_t(cpp_eval_type, cpp_storage_type, do_coarsen)


    print('━━ ', operation_to_perform, ' factorisation ( rtol = ', rtol, ', atol = ', atol, ', overwrite=', overwrite, ' )')

    if operation_to_perform == 'LDL':
        t = time()
        hpro_cpp.LDL_factorize(factors_cpp_object, acc, fac_options)
        dt_fac = time() - t
    elif operation_to_perform == 'LU':
        t = time()
        hpro_cpp.LU_factorize(factors_cpp_object, acc, fac_options)
        dt_fac = time() - t

    print('    done in ', dt_fac)
    print('    size of factors = ', factors_cpp_object.byte_size(), ' bytes')

    if operation_to_perform == 'LDL':
        eval_cpp_object = hpro_cpp.LDL_eval_matrix(factors_cpp_object, fac_options)
        eval_inverse_cpp_object = hpro_cpp.LDL_inv_matrix(factors_cpp_object, fac_options)
    elif operation_to_perform == 'LU':
        eval_cpp_object = hpro_cpp.LU_eval_matrix(factors_cpp_object, fac_options)
        eval_inverse_cpp_object = hpro_cpp.LU_inv_matrix(factors_cpp_object, fac_options)

    return FactorizedHMatrix(eval_cpp_object, eval_inverse_cpp_object, factors_cpp_object, A_hmatrix.bct,
                             eval_type, storage_type, do_coarsen, operation_to_perform)




def build_hmatrix_from_scipy_sparse_matrix(A_csc, bct):
    A_csc[1,0] += 1e-14 # Force non-symmetry
    fname = "temp_sparse_matrix.mat"
    savemat(fname, {'A': A_csc})
    hmatrix_cpp_object = hpro_cpp.build_hmatrix_from_sparse_matfile(fname, bct.cpp_object)
    return HMatrix(hmatrix_cpp_object, bct)


def build_product_convolution_hmatrix_2d(WW_mins, WW_maxes, WW_arrays,
                                         FF_mins, FF_maxes, FF_arrays,
                                         row_dof_coords, col_dof_coords,
                                         block_cluster_tree, tol=1e-6):
    '''Builds hmatrix for product-convolution operator based on weighting functions and convolution kernels defined
    on regular grids in boxes. Convolution kernels must be zero-centered. If a convolution is not zero-centered,
    you can make it zero centered by subtracting the center point from the box min and max points.

    :param WW_mins: weighting function box min points.
        list of numpy arrays, len(WW_mins)=num_patches, WW_mins[k].shape=(2,)
    :param WW_maxes: weighting function box max points.
        list of numpy arrays, len(WW_maxes)=num_patches, WW_maxes[k].shape=(2,)
    :param WW_arrays: arrays of weighting function values on grids.
        list of numpy arrays, len(WW_arrays)=num_patches,
        WW_arrays[k].shape = grid shape for kth weighting function
    :param FF_mins: convolution kernel box min points.
        list of numpy arrays, len(FF_mins)=num_patches, FF_mins[k].shape=(2,)
    :param FF_maxes: convolution kernel box max points.
        list of numpy arrays, len(FF_maxes)=num_patches, FF_maxes[k].shape=(2,)
    :param FF_arrays: arrays of convolution kernel values on grids.
        list of numpy arrays, len(FF_arrays)=num_patches,
        FF_arrays[k].shape = grid shape for kth weighting function
    :param row_dof_coords: array of coordinates in physical space corresponding to rows of the matrix
        row_dof_coords.shape = (num_rows, 2)
    :param col_dof_coords: array of coordinates in physical space corresponding to columns of the matrix
        col_dof_coords.shape = (num_cols, 2)
    :param block_cluster_tree: block cluster tree
    :param tol: truncation tolerance for low rank approximation of Hmatrix low rank (admissible) blocks
    :return: hmatrix
    '''
    PC_cpp = hpro_cpp.ProductConvolution2d(WW_mins, WW_maxes, WW_arrays,
                                           FF_mins, FF_maxes, FF_arrays,
                                           row_dof_coords, col_dof_coords)

    PC_coefffn = hpro_cpp.PC2DCoeffFn(PC_cpp)
    hmatrix_cpp_object = hpro_cpp.build_hmatrix_from_coefffn(PC_coefffn, block_cluster_tree.cpp_object, tol)

    return HMatrix(hmatrix_cpp_object, block_cluster_tree)


def h_factorized_solve(iA_factorized, y):
    return hpro_cpp.h_factorized_inverse_matvec(iA_factorized.cpp_object,
                                                iA_factorized.row_ct.cpp_object,
                                                iA_factorized.col_ct.cpp_object, y)


def h_matvec(A_hmatrix, x):
    return hpro_cpp.h_matvec(A_hmatrix.cpp_object, A_hmatrix.row_ct.cpp_object, A_hmatrix.col_ct.cpp_object, x)

def h_rmatvec(A_hmatrix, x):
    return hpro_cpp.h_rmatvec(A_hmatrix.cpp_object, A_hmatrix.row_ct.cpp_object, A_hmatrix.col_ct.cpp_object, x)

# def visualize_hmatrix(A_hmatrix, title):
#     _hpro_cpp.visualize_hmatrix(A_hmatrix.cpp_object, title)

# def visualize_inverse_factors(iA_factorized, title):
#     _hpro_cpp.visualize_hmatrix(iA_factorized.factors_cpp_object, title)


class ClusterTree:
    def __init__(me, ct_cpp_object):
        me._cpp_object = ct_cpp_object

    @property
    def cpp_object(me):
        return me._cpp_object

    def visualize(me, filename):
        hpro_cpp.visualize_cluster_tree(me.cpp_object, filename)


class BlockClusterTree:
    # wrap block cluster tree cpp object to make sure python doesn't delete row and column cluster trees
    # when they go out of scope but are still internally referenced by the block cluster tree cpp object
    def __init__(me, bct_cpp_object, row_ct, col_ct, admissibility_eta=None):
        me._cpp_object = bct_cpp_object

        me._row_ct = row_ct
        me._col_ct = col_ct
        me._admissibility_eta = admissibility_eta

    @property
    def row_ct(me):
        return me._row_ct

    @property
    def col_ct(me):
        return me._col_ct

    @property
    def admissibility_eta(me):
        return me._admissibility_eta

    @property
    def cpp_object(me):
        return me._cpp_object

    def visualize(me, filename):
        hpro_cpp.visualize_block_cluster_tree(me.cpp_object, filename)


def build_block_cluster_tree(row_ct, col_ct, admissibility_eta=2.0):
    bct_cpp_object = hpro_cpp.build_block_cluster_tree(row_ct.cpp_object, col_ct.cpp_object, admissibility_eta)
    return BlockClusterTree(bct_cpp_object, row_ct, col_ct)


def build_cluster_tree_from_pointcloud(points, cluster_size_cutoff=50):
    '''Build cluster tree from a collection of N points in d dimensions

    :param points: numpy array, shape=(N,d)
    :param cluster_size_cutoff: nonnegative int. number of points below which clusters are not subdivided further
    :return: BlockClusterTreeWrapper
    '''
    cpp_object = hpro_cpp.build_cluster_tree_from_dof_coords(points, cluster_size_cutoff)
    return ClusterTree(cpp_object)

build_cluster_tree_from_dof_coords = build_cluster_tree_from_pointcloud


def rational_positive_definite_approximation_method1(A, overwrite=False,
                                                     rtol_inv=1e-2, atol_inv=1e-15,
                                                     rtol_add=1e-10, atol_add=1e-15,
                                                     min_eig_scale=1.0):
    '''Form symmetric positive definite approximation of hmatrix A with eigenvalues in [m, M]
    using rational approximation of the form:
      A_spd = f(A.sym()) = c0*I + c1*A.sym() + c2 * (A.sym() + mu*I)^-1

    Constants c1, c2, mu are chosen such that:
      f(M) = M
      f(0) = h =approx= 0  (slightly bigger than zero to ensure positive definiteness with inexact hmatrix arithmetic)
      f'(0) = 0
      f(m) = min_eig_scale * |m|

    :param A : HMatrix
    :param overwrite : bool. Overwrite A if true. Otherwise, keep A intact and modify a copy of A
    :param rtol_inv : positive float. Relative error tolerance for H-matrix inversion
    :param atol_inv : nonnegative float. Absolute error tolerance for H-matrix inversion
    :param rtol_add : positive float. Relative error tolerance for H-matrix addition
    :param atol_add : nonnegative float. absolute error tolerance for H-matrix addition
    :param min_eig_scale : nonnegative float. scaling factor: f(m) = min_eig_scale * abs(m)
    :return: HMatrix. Positive definite approximation of A
    '''
    if overwrite:
        X1 = A
    else:
        X1 = A.copy()

    X2 = A.T
    ####     current state:    X1 = A
    ##                         X2 = A^T

    h_add(X2, X1, alpha=0.5, beta=0.5, rtol=rtol_add, atol=atol_add, overwrite_B=True)
    # X1.copy_to(X2) # RuntimeError:  in "(TRkMatrix) copy_to" at "src/matrix/TRkMatrix.cc:2639" Error: invalid matrix type (TDenseMatrix)
    X2 = X1.copy() # COPY FOR DEBUGGING
    ####     current state:    X1 = A.sym()
    ####                       X2 = A.sym()

    M = spla.eigsh(X1, 1)[0][0]
    A2_linop = spla.LinearOperator(X1.shape, matvec=lambda x: M * x - X1 * x)
    m = M - spla.eigsh(A2_linop, 1)[0][0]
    print('A.sym(): lambda_min=', m, ', lambda_max=', M)

    # h = rtol_inv * np.abs(m)
    h = 0

    b = np.array([M, h, 0])

    def make_A(mu):
        return np.array([[1, M, 1 / (M + mu)],
                         [1, 0, 1 / (0 + mu)],
                         [0, 1, -1 / mu ** 2]])

    def res(mu):
        c = np.linalg.solve(make_A(mu), b)
        return min_eig_scale * abs(m) - (c[0] + c[1] * m + c[2] / (m + mu))

    soln = root_scalar(res, x0=-1.9 * m, x1=-2.1 * m)
    mu = soln.root

    c = np.linalg.solve(make_A(mu), b)

    X2.add_identity(mu, overwrite=True)
    X2.inv(overwrite=True, rtol=rtol_inv, atol=atol_inv)
    ####     current state:    X1 = A.sym()
    ####                       X2 = (A.sym() + mu*I)^-1

    h_add(X2, X1, alpha=c[2], beta=c[1], rtol=rtol_add, atol=atol_add, overwrite_B=True)
    ####     current state:    X1 = c1*A.sym() + c2(A.sym() + mu*I)^-1
    ####                       X2 = (A.sym() + mu*I)^-1

    X1.add_identity(c[0], overwrite=True)
    ####     current state:    X1 = c0*I + c1*A.sym() + c2*(A.sym() + mu*I)^-1
    ####                       X2 = (A.sym() + mu*I)^-1

    return X1


def rational_positive_definite_approximation_method2(A, overwrite=False,
                                                     rtol_inv=1e-2, atol_inv=1e-15,
                                                     rtol_add=1e-10, atol_add=1e-15):
    '''Form symmetric positive definite approximation of hmatrix A
    using rational approximation of the form:
      A_spd = c1*A.sym() + c2 * (A.sym() + mu*I)^-1

    Constants c1, c2, mu are chosen such that:
      lambda_min(A_spd) = 0,
      lambda_max(A_spd) = lambda_max(A.sym())
      mu is as small as possible, while maintaining positive definiteness

    :param A: HMatrix
    :param overwrite: bool. Overwrite A if true. Otherwise, keep A intact and modify a copy of A
    :return: HMatrix. Positive definite approximation of A
    '''
    if overwrite:
        M1 = A
    else:
        M1 = A.copy()

    M2 = A.T
    ####     current state:    M1 = A
    ##                         M2 = A^T

    h_add(M2, M1, alpha=0.5, beta=0.5, rtol=rtol_add, atol=atol_add, overwrite_B=True)
    M2 = M1.copy() # M1.copy_to(M2) # COPY FOR DEBUGGING
    ####     current state:    M1 = A.sym()
    ####                       M2 = A.sym()

    lambda_max = spla.eigsh(M1, 1)[0][0]
    A2_linop = spla.LinearOperator(M1.shape, matvec=lambda x: lambda_max * x - M1 * x)
    lambda_min = lambda_max - spla.eigsh(A2_linop, 1)[0][0]
    print('A.sym(): lambda_min=', lambda_min, ', lambda_max=', lambda_max)

    # zerpoint: location where rational function crosses zero.
    # Slightly smaller than lambda_min to ensure positive definiteness with inexact hmatrix arithmetic
    zeropoint = np.abs(lambda_min) * (1. + 2*rtol_inv)
    mu = 2.0 * zeropoint
    gamma = zeropoint*(mu - zeropoint)
    c1 = (1. / (1. + gamma / (lambda_max*(lambda_max + mu))))
    c2 = c1 * gamma

    M2.add_identity(mu, overwrite=True)
    M2.inv(overwrite=True, rtol=rtol_inv, atol=atol_inv)
    ####     current state:    M1 = A.sym()
    ####                       M2 = (A.sym() + mu*I)^-1

    h_add(M2, M1, alpha=c2, beta=c1, rtol=rtol_add, atol=atol_add, overwrite_B=True)
    ####     current state:    M1 = c1*A.sym() + c2(A.sym() + mu*I)^-1
    ####                       M2 = (A.sym() + mu*I)^-1

    return M1


def rational_positive_definite_approximation_low_rank_method(A,
                                                             cutoff=0.0,
                                                             block_size=20,
                                                             max_rank=500,
                                                             display=True,
                                                             overwrite=False,
                                                             rtol=_DEFAULT_RTOL,
                                                             atol=_DEFAULT_ATOL):
    '''Form symmetric positive definite approximation of hmatrix A
    uses low rank deflation to flip negative eigenvalues less than cutoff.
    '''
    cutoff = -np.abs(cutoff)

    A_plus = A.sym(rtol=rtol, atol=atol, overwrite=overwrite)

    update_rank=0
    while update_rank < max_rank:
        print('getting negative eigs')
        min_eigs, min_evecs = spla.eigsh(A_plus.as_linear_operator(), block_size, which='SA')
        negative_inds = min_eigs < 0
        ee_neg = min_eigs[negative_inds]
        U_neg = min_evecs[:, negative_inds]
        update_rank += len(ee_neg)

        X_update = U_neg
        Y_update = np.dot(np.diag(-2 * ee_neg), U_neg.T)

        print('hmatrix low rank update')
        A_plus.low_rank_update(X_update, Y_update, overwrite=True, rtol=rtol, atol=atol)

        if display:
            print('update_rank=', update_rank)
            print('min_eigs=', min_eigs)
            print('cutoff=', cutoff)
        if (np.max(min_eigs) > cutoff):
            print('negative eigs smaller than cutoff. Good.')
            break

    return A_plus

###

# f_2(x) = 1/(1+x^2)
# f_2(A) = (I + A*A)^-1
# f_32(x) = 1/(1+x^32)
# f_32(A) = (I + A*A*...*A)^-1
# A^2 = A*A
# A^4 = A^2 * A^2
# A^8 = A^4 * A^4
# ..
# A^32 = A^16 * A^16
# A * f_32(A) =approx= P diag(ee_plus) * P^T
# A * P = P * diag(ee)
# ee_plus(k) = ee(k) if ee(k) >=0 or 0 if ee(k) < 0
# 1 / (1 + x^(2^k))

def make_hmatrix_spd_hackbusch_kress_2007(A_hmatrix, k=2, rtol=_DEFAULT_RTOL, atol=_DEFAULT_ATOL, display_progress=False,
                                          a_factor=1.5, b_factor=0.0):
    # Hackbusch, Wolfgang, and Wendy Kress. "A projection method for the computation of inner eigenvalues using high degree rational operators." Computing 81.4 (2007): 259-268.
    if display_progress:
        print('making hmatrix spd')
        print('symmetrizing')
    A_hmatrix = A_hmatrix.sym()

    if display_progress:
        print('getting smallest eigenvalue with Lanczos')
    ee_SA, _ = spla.eigsh(A_hmatrix.as_linear_operator(), k=1, which='SA')
    lambda_min = np.min(ee_SA)
    if display_progress:
        print('lambda_min=', lambda_min)

    a = a_factor * lambda_min
    b = b_factor * lambda_min

    if display_progress:
        scaling_at_lambda_min = 1. / (1.0 + ((2.0*lambda_min - (b + a)) / (b - a)) ** (2 ** k))
        print('scaling_at_lambda_min=', scaling_at_lambda_min)

        scaling_at_zero = 1. / (1.0 + (-(b+a)/(b-a))**(2**k))
        print('scaling_at_zero=', scaling_at_zero)

    if display_progress:
        print('Setting up operator T = (2*A - (b+a) I) / (b-a)')

    T = A_hmatrix.copy()
    T = (T * 2.0).add_identity(s=-(b + a)) * (1.0 / (b - a))

    if display_progress:
        print('computing T^(2^k)')
    for ii in range(k):
        if display_progress:
            print('computing T^(2^'+str(ii+1)+') = T^(2^'+str(ii)+') * T^(2^'+str(ii)+')')
        T = h_mul(T, T,
                  rtol=rtol, atol=atol,
                  display_progress=display_progress)

    if display_progress:
        print('computing negative spectral projector Pi_minus = I / (I + T^(2^k))')
    Pi_minus = T.add_identity().inv(rtol=rtol, atol=atol, display_progress=display_progress)

    if display_progress:
        print('computing absolute value projector Pi = I - 2*Pi_minus')
    Pi = (Pi_minus * (-2.0)).add_identity()

    if display_progress:
        print('computing A_plus = Pi * A')
    A_plus = h_mul(Pi, A_hmatrix,
                   rtol=rtol, atol=atol,
                   display_progress=display_progress).sym()

    return A_plus


def make_shifted_factorization(
        A: HMatrix,
        B: HMatrix,
        mu: float,
        display=False,
        rtol=_DEFAULT_RTOL,
) -> FactorizedInverseHMatrix:
    assert(mu > 0.0)
    A_plus_muB = h_add(A, B, 1.0, mu)
    if display:
        print('Factorizing A+mu*B')
    A_plus_muB_fac = A_plus_muB.factorized_inverse(rtol=rtol, overwrite=False)
    if display:
        print('Done factorizing A+mu*B')
    return A_plus_muB_fac


def negative_eigenvalues_of_hmatrix_pencil(
        A: HMatrix, # shape=(N,N), symmetric
        B: HMatrix, # shape=(N,N), symmetric positive definite
        range_min: float, # range_min < range_max < 0. range_min can be None
        range_max: float,
        prior_dd: np.ndarray=None,
        prior_V: np.ndarray=None,
        B_fac: FactorizedInverseHMatrix=None,
        save_intermediate_factorizations: bool=True,
        display: bool=False,
        tol: float=1e-8,
        additional_options: typ.Dict[str, typ.Any]=None,
) -> typ.Tuple[np.ndarray, np.ndarray, typ.List[float], typ.List[FactorizedInverseHMatrix]]:
    N = A.shape[0]
    assert(A.shape == (N, N))
    assert(B.shape == (N, N))
    assert(range_max < 0.0)
    def printmaybe(*args, **kwargs):
        if display:
            print(*args, **kwargs)

    if additional_options is None:
        additional_options = dict()

    if (range_min is None) and (B_fac is None):
        printmaybe('Factorizing B')
        B_fac = B.factorized_inverse(rtol=tol, overwrite=False)
        printmaybe('Done factorizing B')

    shifts: typ.List[float] = []
    factorized_shifted_matrices: typ.List[FactorizedInverseHMatrix] = []
    def make_shifted_solver(shift):
        A_minus_shiftB_fac = make_shifted_factorization(A, B, -shift, display=display, rtol=tol)
        if save_intermediate_factorizations:
            shifts.append(shift)
            factorized_shifted_matrices.append(A_minus_shiftB_fac)
        return A_minus_shiftB_fac.matvec

    dd, V = negative_eigenvalues_of_matrix_pencil(
        A.matvec, B.matvec, make_shifted_solver, N,
        range_min, range_max,
        prior_dd=prior_dd, prior_V=prior_V,
        solve_B=B_fac.matvec,
        tol=tol, display=display, **additional_options)

    return dd, V, shifts, factorized_shifted_matrices


@dataclass
class HMatrixShiftedInverseInterpolator:
    '''
    A is a symmetric NxN HMatrix
    B is a symmetric positive definite HMatrix
    shifted_factorizations[ii]: factorization of A+mu[ii]*B
    dd and U are eigenvalues and eigenvectors of matrix pencil (A, B) that we deflate
    BU = B @ U
    gamma is the amount of deflation applied (-1.0: set chosen eigs to zero, -2.0: flip chosen eigs)
    '''
    A: HMatrix
    B: HMatrix
    mus: typ.List[float]
    shifted_factorizations: typ.List[FactorizedInverseHMatrix]
    B_fac: FactorizedInverseHMatrix
    dd: np.ndarray
    BU: np.ndarray # B @ U
    spectrum_lower_bound: float
    gamma: float
    fac_rtol: float
    check_rtol: float
    display: bool

    def __init__(me,
                 A: HMatrix,
                 B: HMatrix,
                 mus: typ.List[float]=None,
                 shifted_factorizations: typ.List[FactorizedInverseHMatrix]=None,
                 B_fac: FactorizedInverseHMatrix=None,
                 dd: np.ndarray = None,
                 BU: np.ndarray = None,
                 LM_eig: float = None,
                 spectrum_lower_bound: float = None,
                 gamma: float=-2.0, # -2.0 for flipping negative eigs, -1.0 to set them to zero
                 fac_rtol: float = 1e-10,
                 check_rtol: float = 1e-6,
                 display: bool=False,
                 ):
        assert(check_rtol > 0.0)
        me.check_rtol = check_rtol
        assert(fac_rtol > 0.0)
        assert(check_rtol < fac_rtol)
        me.fac_rtol = fac_rtol
        me.display = display
        assert(gamma <= 0.0)
        me.gamma = gamma

        me.A = A
        me.B = B
        assert (me.A.shape == (me.N, me.N))
        assert (me.B.shape == (me.N, me.N))

        # check symmetry of A and B
        u = np.random.randn(me.N)
        v = np.random.randn(me.N)
        tA1 = np.dot(me.A.matvec(u), v)
        tA2 = np.dot(u, me.A.matvec(v))
        assert(np.abs(tA2 - tA1) <= me.check_rtol * (np.abs(tA1) + np.abs(tA2))) # A is symmetric

        tB1 = np.dot(me.B.matvec(u), v)
        tB2 = np.dot(u, me.B.matvec(v))
        assert (np.abs(tB2 - tB1) <= me.check_rtol * (np.abs(tB1) + np.abs(tB2)))  # B is symmetric

        # check correctness of shifted factorizations
        me.mus = mus if mus is not None else []
        me.shifted_factorizations = shifted_factorizations if shifted_factorizations is not None else []
        assert(len(me.mus) == len(me.shifted_factorizations))

        for mu, fac in zip(me.mus, me.shifted_factorizations):
            me.check_shifted_factorization(mu, fac)

        # Make B factorization if it is not supplied. Check it's correctness
        if B_fac is None:
            me.B_fac = B.factorized_inverse(rtol=me.fac_rtol, overwrite=False)
        else:
            me.B_fac = B_fac
        assert(B_fac.shape == (me.N, me.N))
        x = np.random.randn(me.N)
        x2 = me.B_fac.matvec(me.B.matvec(x))
        assert(np.linalg.norm(x2 - x) <= me.check_rtol * np.linalg.norm(x)) # B_fac is correct

        # Check correctness of supplied generalized eigenvalues and eigenvectors
        if dd is None:
            me.dd = np.zeros((0,))
        else:
            me.dd = dd
        assert(me.dd.shape == (me.k,))
        assert(np.all(dd <= 0.0))

        if BU is None:
            me.BU = np.zeros((me.N,0))
        else:
            me.BU = BU
        assert(me.BU.shape == (me.N, me.k))
        me.check_generalized_eigenproblem_correctness(me.dd, me.BU)

        if LM_eig is None:
            me.printmaybe('Computing largest magnitude eigenvalue of (A, B)')
            me.LM_eig = spla.eigsh(spla.LinearOperator((me.N, me.N), matvec=me.A.matvec),
                                   1,
                                   M=spla.LinearOperator((me.N, me.N), matvec=me.B.matvec),
                                   Minv=spla.LinearOperator((me.N, me.N), matvec=me.B_fac.matvec),
                                   which='LM', return_eigenvectors=False,
                                   tol=me.check_rtol)[0]
            me.printmaybe('LM_eig=', me.LM_eig)
        else:
            me.LM_eig = LM_eig

        if spectrum_lower_bound is None:
            spectrum_lower_bound = -me.LM_eig
        assert(spectrum_lower_bound < 0.0)
        me.spectrum_lower_bound = spectrum_lower_bound

    def printmaybe(me, *args, **kwargs):
        if me.display:
            print(*args, **kwargs)

    def check_shifted_factorization(me, mu, shifted_factorization):
        assert(shifted_factorization.shape == (me.N, me.N))
        x = np.random.randn(me.N)
        b = me.A.matvec(x) + mu * me.B.matvec(x)
        x2 = shifted_factorization.matvec(b)
        assert(np.linalg.norm(x2 - x) <= me.check_rtol * np.linalg.norm(x)) # shifted factorization is correct

    def check_generalized_eigenproblem_correctness(me, check_dd: np.ndarray, check_BU: np.ndarray):
        # Require:
        # U.T @ A @ U = diag(dd)    Equation 1
        # U.T @ B @ U = I           Equation 2
        check_k = len(check_dd)
        assert(check_dd.shape == (check_k,))
        assert(check_BU.shape == (me.N, check_k))
        if check_k  > 0:
            x1 = np.random.randn(check_k)
            x2 = np.random.randn(check_k)
            z1 = me.B_fac.matvec(check_BU @ x1)
            z2 = me.B_fac.matvec(check_BU @ x2)
            E1_left = np.dot(z1, me.A.matvec(z2))
            E1_right = np.sum(x1 * check_dd * x2)
            assert(np.abs(E1_left - E1_right) < me.check_rtol * (np.abs(E1_left) + np.abs(E1_right))) # U.T @ A @ U = diag(dd)
            E2_left = np.dot(z1, me.B.matvec(z2))
            E2_right = np.sum(x1 * x2)
            assert(np.abs(E2_left - E2_right) < me.check_rtol * (np.abs(E2_left) + np.abs(E2_right)))  # U.T @ B @ U = I

    @cached_property
    def N(me) -> int:
        return me.A.shape[0]

    @cached_property
    def k(me) -> int:
        return len(me.dd)

    def apply_shifted_deflated(me, x: np.ndarray, mu: float) -> np.ndarray:
        '''x -> (A + gamma*V @ diag(dd) @ V.T + mu0*B) @ x'''
        assert(mu > 0.0)
        return me.A.matvec(x) + mu * me.B.matvec(x) + me.gamma * me.BU @ (me.dd * (me.BU.T @ x))

    def solve_shifted_deflated_with_known_mu(me, b: np.ndarray, mu_ind: int) -> np.ndarray:
        return solve_shifted_deflated(
            b, me.shifted_factorizations[mu_ind].matvec, -me.mus[mu_ind], me.gamma, me.dd, me.BU)

    def solve_shifted_deflated_preconditioner(me, b: np.ndarray, mu: float, display=True) -> np.ndarray:
        '''b -> c0*(A + gamma*V @ diag(dd) @ V.T + mu0*B)^-1 @ b + c1*(A + gamma*V @ diag(dd) @ V.T + mu0*B)^-1 @ b
             =approx= (A + gamma*V @ diag(dd) @ V.T + mu*B)^-1 @ b, where:
        mu0 < mu < mu1
        c0 and c1 are chosen so that the approximation is exact on the largest end of the spectrum of A,
        the zero end of the spectrum.'''
        assert(mu > 0.0)
        assert(b.shape == (me.N,))
        known_shifted_deflated_solvers = [
            lambda b, _mu=mu: me.solve_shifted_deflated_with_known_mu(b, _mu)
            for mu in me.mus]
        return shifted_inverse_interpolation_preconditioner(
            b, mu, me.mus, known_shifted_deflated_solvers,
            np.abs(me.LM_eig), display=display)

    def insert_new_mu(me, new_mu: float, new_fac: FactorizedInverseHMatrix=None) -> 'HMatrixShiftedInverseInterpolator':
        assert(new_mu > 0.0)
        if new_fac is None:
            new_fac = make_shifted_factorization(me.A, me.B, new_mu, rtol=me.fac_rtol, display=me.display)
        me.check_shifted_factorization(new_mu, new_fac)
        me.mus.append(new_mu)
        me.shifted_factorizations.append(new_fac)

    def insert_new_deflation(me, new_dd: np.ndarray, new_BU: np.ndarray):
        new_k = len(new_dd)
        assert(new_dd.shape == (new_k,))
        assert(new_BU.shape == (me.N, new_k))
        proposed_dd = np.concatenate([me.dd, new_dd])
        proposed_BU = np.hstack([me.BU, new_BU])
        me.check_generalized_eigenproblem_correctness(proposed_dd, proposed_BU)
        me.dd = proposed_dd
        me.BU = proposed_BU

    def deflate_more(me, new_spectrum_lower_bound: float, **additional_options):
        assert(new_spectrum_lower_bound < 0.0)
        if new_spectrum_lower_bound > me.spectrum_lower_bound:
            new_dd, new_BU, new_shifts, new_facs = negative_eigenvalues_of_hmatrix_pencil(
                me.A, me.B, me.spectrum_lower_bound, new_spectrum_lower_bound,
                prior_dd=me.dd, prior_V=me.BU,
                B_fac=me.B_fac,
                save_intermediate_factorizations=True,
                display=me.display, tol=me.fac_rtol,
                additional_options=additional_options)

            new_mus = list(-np.array(new_shifts))
            for mu, fac in zip(new_mus, new_facs):
                me.insert_new_mu(mu, fac)

            me.insert_new_deflation(new_dd, new_BU)


def make_shifted_hmatrix_inverse_interpolator(
        A: HMatrix,
        B: HMatrix,
        mu_min: float,
        mu_max: float,
        LM_eig: float=None,
        mu_spacing_factor: float = _MU_SPACING_FACTOR, # e.g., 50.0
        known_mus: typ.List[float] = None,
        known_shifted_factorizations: typ.List[FactorizedInverseHMatrix] = None,
        deflation_dd: np.ndarray = None,
        deflation_V: np.ndarray = None,
        rtol: float = _DEFAULT_RTOL,
        boundary_mu_rtol = _BOUNDARY_MU_RTOL, # e.g., 0.1
        gamma: float = -1.0,
        display=True,
        perturb_mu_factor: float=1e-3,
) -> HMatrixShiftedInverseInterpolator:
    '''Make object that approximates action of (A + gamma*V @ diag(dd) @ V.T + mu*B)^-1
    at arbitrary mu in [mu_min, mu_max] via weighted sum of known factorizations
    of operators A + gamma*V @ diag(dd) @ V.T + mu_k*B.
    Values mu_k are chosen to fill the range [mu_min, mu_max] to within factor mu_spacing_factor
    I.e., if mu_i are ordered in increasing order, then mu_(i+1) <= mu_spacing_factor*mu_i.
    Ensures that there are values of mu_k near mu_min and mu_max to within factor boundary_mu_rtol.
    I.e., there exists mu_k such that |mu_k - mu_min| < boundary_mu_rtol*|mu_min|,
    and similar for mu_max.
    Requirements:
        A symmetric
        B symmetric positive definite
        dd, V are generalized eigenvalues/eigenvectors of (A, B)'''
    assert (0.0 < mu_min)
    assert (mu_min <= mu_max)
    N = A.shape[0]
    assert (A.shape == (N, N))
    assert (B.shape == (N, N))
    assert (rtol > 0.0)
    assert (mu_spacing_factor > 1.0)
    assert(boundary_mu_rtol >= 0.0)
    assert(np.abs(perturb_mu_factor) < 1.0)

    mu_min = (1.0 + perturb_mu_factor*(np.random.rand() - 0.5)) * mu_min
    mu_max = (1.0 + perturb_mu_factor*(np.random.rand() - 0.5)) * mu_max

    if LM_eig is None:
        B_fac = B.factorized_inverse(rtol=rtol,overwrite=False)
        LM_eig = spla.eigsh(spla.LinearOperator((N, N), matvec=A.matvec),
                            1,
                            M=spla.LinearOperator((N, N), matvec=B.matvec),
                            Minv=spla.LinearOperator((N, N), matvec=B_fac.matvec),
                            which='LM', return_eigenvectors=False,
                            tol=rtol)[0]

    if deflation_dd is None:
        assert(deflation_V is None)
        deflation_dd = np.zeros((0,))
        deflation_V = np.zeros((N, 0))

    if known_mus is None:
        assert(known_shifted_factorizations is None)
        known_mus = []
        known_shifted_factorizations = []
    else:
        assert (len(known_mus) == len(known_shifted_factorizations))
        known_mus = known_mus.copy() # Don't modify original list
        known_shifted_factorizations = known_shifted_factorizations.copy()

    for mu in known_mus:
        assert (mu > 0.0)

    # Get new mus to fill in gaps
    dont_have_mu_min = True
    if len(known_mus) > 0:
        dont_have_mu_min = np.min(np.abs(np.array(known_mus) - mu_min)) > boundary_mu_rtol * mu_min

    dont_have_mu_max = True
    if len(known_mus) > 0:
        dont_have_mu_max = np.min(np.abs(np.array(known_mus) - mu_max)) > boundary_mu_rtol * mu_max

    if dont_have_mu_min:
        if display:
            print('making shifted factorization A + mu_min*B, mu_min=', mu_min)
        fac = make_shifted_factorization(A, B, mu_min, rtol=rtol, display=display)
        known_mus.append(mu_min)
        known_shifted_factorizations.append(fac)
    if dont_have_mu_max:
        if display:
            print('making shifted factorization A + mu_max*B, mu_max=', mu_max)
        fac = make_shifted_factorization(A, B, mu_max, rtol=rtol, display=display)
        known_mus.append(mu_max)
        known_shifted_factorizations.append(fac)

    # Algorithm sub-optimal; looping and sorting way more than needed
    sort_inds = list(np.argsort(known_mus).reshape(-1))
    known_mus = [known_mus[ind] for ind in sort_inds]
    known_shifted_factorizations = [known_shifted_factorizations[ind] for ind in sort_inds]
    ii = 0
    while ii + 1 < len(known_mus):
        mu_low = known_mus[ii]
        mu_high = known_mus[ii + 1]
        if mu_high / mu_low > mu_spacing_factor and mu_low < mu_max:
            mu_mid = np.exp(0.5 * (np.log(mu_low) + np.log(mu_high)))
            if mu_mid / mu_low > mu_spacing_factor:
                mu_mid = mu_low * mu_spacing_factor
            if display:
                print('making A + mu*B factorization. (mu_low, mu, mu_high)=', (mu_low, mu_mid, mu_high))
            fac = make_shifted_factorization(A, B, mu_mid, rtol=rtol, display=display)
            known_mus.append(mu_mid)
            known_shifted_factorizations.append(fac)

            sort_inds = list(np.argsort(known_mus).reshape(-1))
            known_mus = [known_mus[ind] for ind in sort_inds]
            known_shifted_factorizations = [known_shifted_factorizations[ind] for ind in sort_inds]
        ii += 1

    return HMatrixShiftedInverseInterpolator(A, B, LM_eig, known_mus,
                                             known_shifted_factorizations,
                                             deflation_dd, deflation_V, gamma)


def deflate_negative_eigs_then_make_shifted_hmatrix_inverse_interpolator(
        A: HMatrix,
        B: HMatrix,
        mu_min: float,
        mu_max: float,
        mu_spacing_factor: float = _MU_SPACING_FACTOR,
        rtol: float = _DEFAULT_RTOL,
        boundary_mu_rtol: float=_BOUNDARY_MU_RTOL, # e.g. 0.1
        gamma: float = -2.0,
        threshold: float=-0.5,
        negative_eigenvalue_finder_options: typ.Dict[str, typ.Any]=None,
        display: bool=True,
) -> HMatrixShiftedInverseInterpolator:
    assert(0.0 < mu_min)
    assert(mu_min < mu_max)
    assert(mu_spacing_factor > 1.0)
    assert(rtol > 0.0)
    assert(boundary_mu_rtol > 0.0)
    assert(threshold < 0.0)
    assert(gamma < 0.0)

    if negative_eigenvalue_finder_options is None:
        negative_eigenvalue_finder_options = dict()

    B_min = h_scale(B, mu_min)
    dd_min, V_min, shifts_min, factorized_shifted_matrices = negative_eigenvalues_of_hmatrix_pencil(
        A, B_min, None, threshold,
        save_intermediate_factorizations=True,
        display=display,
        tol=rtol,
        additional_options=negative_eigenvalue_finder_options,
    )

        # negative_eigenvalues_of_hmatrix_pencil(
        # A, B_min, None, threshold,
        # save_intermediate_factorizations=save_intermediate_factorizations,
        # threshold=threshold, sigma_factor=np.sqrt(mu_spacing_factor), chunk_size=chunk_size,
        # tol=rtol, ncv_factor=ncv_factor, lanczos_maxiter=lanczos_maxiter, display=display,
        # shifted_preconditioner_only=shifted_preconditioner_only)
    # dd_min, V_min, shifts_min, factorized_shifted_matrices, LM_eig_min = negative_eigenvalues_of_hmatrix_pencil(
    #     A, B_min, threshold=threshold)

    V = V_min / np.sqrt(mu_min) #* np.sqrt(mu_min)
    dd = dd_min * mu_min
    LM_eig = LM_eig_min * mu_min
    known_mus = [-shift * mu_min for shift in shifts_min]
    if display:
        print('known_mus=', known_mus)

    return make_shifted_hmatrix_inverse_interpolator(
        A, B, mu_min, mu_max, LM_eig,
        mu_spacing_factor=mu_spacing_factor,
        known_mus=known_mus,
        known_shifted_factorizations=factorized_shifted_matrices,
        deflation_dd=dd,
        deflation_V=V,
        rtol=rtol,
        boundary_mu_rtol=boundary_mu_rtol,
        gamma=gamma,
        display=display,
    )



make_hlibpro_low_rank_matrix = hpro_cpp.make_hlibpro_low_rank_matrix # X = A @ B.T
make_permuted_hlibpro_low_rank_matrix = hpro_cpp.make_permuted_hlibpro_low_rank_matrix

