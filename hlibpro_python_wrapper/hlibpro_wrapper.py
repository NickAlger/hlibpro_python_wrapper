import numpy as np
import scipy.sparse.linalg as spla
from scipy.optimize import root_scalar
from scipy.io import savemat
from time import time
from . import hlibpro_bindings as hpro_cpp

default_rtol = 1e-7
default_atol = 1e-12


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

    def sym(me, rtol=default_rtol, atol=default_atol, overwrite=False):
        if overwrite:
            A_sym = h_add(me.T, me, alpha=0.5, beta=0.5, rtol=rtol, atol=atol, overwrite_B=True)
        else:
            A_sym = h_add(me, me.T, alpha=0.5, beta=0.5, rtol=rtol, atol=atol, overwrite_B=True)
        # A_sym._set_symmetric()
        return A_sym

    def spd(me, **kwargs):
        return rational_positive_definite_approximation_low_rank_method(me, **kwargs)
        # return rational_positive_definite_approximation_method1(me, **kwargs)

    def _set_symmetric(me):
        me._cpp_object.set_symmetric()

    def _set_nonsym(me):
        me._cpp_object.set_nonsym()

    def matvec(me, x):
        return h_matvec(me, x)

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

    def __sub__(me, other, rtol=default_rtol, atol=default_atol):
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

        else:
            raise RuntimeError('cannot right multiply HMatrix with ' + str(other.type))

    def __matmul__(me, other):
        return me.__mul__(me, other)

    def __rmatmul__(me, other):
        return me.__rmul__(me, other)

    # def __imatmul__(me, other):
    #     return me.__imul__(me, other)

    def inv(me, rtol=default_rtol, atol=default_atol,
            overwrite=False, display_progress=True,
            diag_type='general_diag', storage_type='store_normal', do_coarsen=False):
        return h_inv(me, rtol=rtol, atol=atol,
                     overwrite=overwrite, display_progress=display_progress,
                     diag_type=diag_type, storage_type=storage_type, do_coarsen=do_coarsen)

    def factorized_inverse(me, rtol=default_rtol, atol=default_atol, overwrite=False):
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
        return spla.LinearOperator(me.shape, matvec=me.matvec)

    def low_rank_update(me, X, Y, overwrite=False, rtol=default_rtol, atol=default_atol): #A -> A + X*Y
        XY = hpro_cpp.make_permuted_hlibpro_low_rank_matrix(np.copy(X), np.copy(Y.T), me.row_ct.cpp_object, me.col_ct.cpp_object)

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
                   rtol=default_rtol,
                   atol=default_atol):
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

        YtX = np.dot(Y.T, X)
        ee, P = np.linalg.eigh(YtX) # note: symmetric because Y = A*X
        if np.any(ee < 0):
            print('warning: negative directions detected in DFP update')
        if force_positive_definite:
            iee = 1./np.abs(ee)
        else:
            iee = 1./ee
        iYtX = np.dot(P, np.dot(np.diag(iee), P.T))
        Q = np.dot(Y, iYtX)

        Y0 = np.zeros((me.shape[0], X.shape[1]))
        for k in range(X.shape[1]):
            Y0[:,k] = me.matvec(X[:,k].copy())

        L = np.hstack([np.dot(Q, np.dot(X.T, Y0)) - Y0, Q])
        R = np.hstack([Q, Y - Y0]).T

        print('doing low rank update')
        A1 = me.low_rank_update(L, R, overwrite=overwrite, rtol=rtol, atol=atol)

        if check_correctness:
            Y2 = np.zeros(Y.shape)
            for k in range(X.shape[1]):
                Y2[:,k] = A1.matvec(X[:,k])
            err_dfp = np.linalg.norm(Y2 - Y) / np.linalg.norm(Y)
            print('err_dfp=', err_dfp, ', rtol=', rtol, ', atol=', atol)

        return A1

    def broyden_update(me, X, Y,
                       overwrite=False,
                       check_correctness=True,
                       rtol=default_rtol,
                       atol=default_atol):
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
                   rtol=default_rtol,
                   atol=default_atol):
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


def h_add(A_hmatrix, B_hmatrix, alpha=1.0, beta=1.0, rtol=default_rtol, atol=default_atol, overwrite_B=False):
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


def h_mul(A_hmatrix, B_hmatrix, alpha_A_B_hmatrix=None, alpha=1.0, rtol=default_rtol, atol=default_atol, display_progress=True):
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


def h_factorized_inverse(A_hmatrix, rtol=default_rtol, atol=default_atol, overwrite=False):
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

def h_inv(A_hmatrix, rtol=default_rtol, atol=default_atol,
          overwrite=False, display_progress=True,
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
        inv_options = hpro_cpp.fac_options_t(cpp_diag_type, cpp_storage_type, do_coarsen)

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


def h_ldl(A_hmatrix, rtol=default_rtol, atol=default_atol,
          overwrite=False, display_progress=True,
          eval_type='block_wise', storage_type='store_normal', do_coarsen=False):
    return _factorize_h_matrix(A_hmatrix, 'LDL', rtol, atol,
                               overwrite, display_progress,
                               eval_type, storage_type, do_coarsen)


def h_lu(A_hmatrix, rtol=default_rtol, atol=default_atol,
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
                                                             rtol=default_rtol,
                                                             atol=default_atol):
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


make_hlibpro_low_rank_matrix = hpro_cpp.make_hlibpro_low_rank_matrix # X = A @ B.T
make_permuted_hlibpro_low_rank_matrix = hpro_cpp.make_permuted_hlibpro_low_rank_matrix

