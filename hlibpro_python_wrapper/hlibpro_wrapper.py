import numpy as np
import scipy.sparse.linalg as spla
from scipy.io import savemat
from time import time
from . import hlibpro_bindings as _hpro_cpp

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

    def copy(me):
        return HMatrix(_hpro_cpp.copy_TMatrix(me.cpp_object), me.bct)

    def copy_struct(me):
        return HMatrix(_hpro_cpp.copy_struct_TMatrix(me.cpp_object), me.bct)

    def copy_to(me, other):
        _hpro_cpp.copy_TMatrix_into_another_TMatrix(me.cpp_object, other.cpp_object)
        other._bct = me.bct

    def transpose(me, overwrite=False):
        if overwrite:
            me._cpp_object.transpose()
            return me
        else:
            transposed_cpp_object = _hpro_cpp.copy_TMatrix(me.cpp_object)
            transposed_cpp_object.transpose()
            return HMatrix(transposed_cpp_object, me.bct)

    @property
    def T(me):
        return me.transpose()

    def sym(me, rtol=default_rtol, atol=default_atol):
        A_sym = h_add(me, me.T, alpha=0.5, beta=0.5, rtol=rtol, atol=atol, overwrite_B=True)
        A_sym._set_symmetric()
        return A_sym

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

    def __imul__(me, other):
        if isinstance(other, HMatrix):
            return h_mul(me, other, alpha_A_B_hmatrix=me)

        if isinstance(other, float) or isinstance(other, np.number):
            return h_scale(me, other, overwrite_A=True)

        else:
            raise RuntimeError('cannot multiply HMatrix with ' + str(other.type))

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

    def __imatmul__(me, other):
        return me.__imul__(me, other)

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

    def visualize(me, filename):
        _hpro_cpp.visualize_hmatrix(me.cpp_object, filename)


def add_identity_to_hmatrix(A_hmatrix, s=1.0, overwrite=False):
    if overwrite:
        A_plus_alpha_I = A_hmatrix
    else:
        A_plus_alpha_I = A_hmatrix.copy()
    _hpro_cpp.add_identity_to_hmatrix(A_plus_alpha_I.cpp_object, s)
    return A_plus_alpha_I


def h_add(A_hmatrix, B_hmatrix, alpha=1.0, beta=1.0, rtol=default_rtol, atol=default_atol, overwrite_B=False):
    # C = A + alpha * B to tolerance given by truncation accuracy object acc
    acc = _hpro_cpp.TTruncAcc(relative_eps=rtol, absolute_eps=atol)
    if overwrite_B:
        alpha_A_plus_beta_B_hmatrix = B_hmatrix
    else:
        alpha_A_plus_beta_B_hmatrix = B_hmatrix.copy()
    _hpro_cpp.add(alpha, A_hmatrix.cpp_object, beta, alpha_A_plus_beta_B_hmatrix.cpp_object, acc)
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
    acc = _hpro_cpp.TTruncAcc(relative_eps=rtol, absolute_eps=atol)
    if alpha_A_B_hmatrix is None:
        # AB_admissibility_eta = np.min([A_hmatrix.admissibility_eta, B_hmatrix.admissibility_eta])
        # AB_bct = build_block_cluster_tree(A_hmatrix.row_ct, B_hmatrix.col_ct, AB_admissibility_eta)
        alpha_A_B_hmatrix = A_hmatrix.copy_struct()

    if display_progress:
        _hpro_cpp.multiply_with_progress_bar(alpha, _hpro_cpp.apply_normal, A_hmatrix.cpp_object,
                                             _hpro_cpp.apply_normal, B_hmatrix.cpp_object,
                                             0.0, alpha_A_B_hmatrix.cpp_object, acc)
    else:
        _hpro_cpp.multiply_without_progress_bar(alpha, _hpro_cpp.apply_normal, A_hmatrix.cpp_object,
                                                _hpro_cpp.apply_normal, B_hmatrix.cpp_object,
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
    acc = _hpro_cpp.TTruncAcc(relative_eps=rtol, absolute_eps=atol)
    if overwrite:
        factors_cpp_object = A_hmatrix.cpp_object
    else:
        factors_cpp_object = _hpro_cpp.copy_TMatrix(A_hmatrix.cpp_object)
    cpp_object = _hpro_cpp.factorize_inv_with_progress_bar(factors_cpp_object, acc)
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
        return _hpro_cpp.h_factorized_inverse_matvec(me.eval_inverse_cpp_object,
                                                     me.row_ct.cpp_object,
                                                     me.col_ct.cpp_object, y)

    def apply(me, x):
        return _hpro_cpp.h_factorized_matvec(me.eval_cpp_object,
                                             me.row_ct.cpp_object,
                                             me.col_ct.cpp_object, x)

    def as_linear_operator(me, inverse=False):
        if inverse:
            return spla.LinearOperator(me.shape, matvec=me.solve)
        else:
            return spla.LinearOperator(me.shape, matvec=me.apply)

    def visualize(me, filename):
        _hpro_cpp.visualize_hmatrix(me.factors_cpp_object, filename)

    def split(me):
        if me.eval_type == 'point_wise':
            cpp_eval_type = _hpro_cpp.eval_type_t.point_wise
        elif me.eval_type == 'block_wise':
            cpp_eval_type = _hpro_cpp.eval_type_t.block_wise
        else:
            raise RuntimeError('eval_type must be point_wise or block_wise. eval_type=', me.eval_type)

        if me.storage_type == 'store_normal':
            cpp_storage_type = _hpro_cpp.storage_type_t.store_normal
        elif me.storage_type == 'store_inverse':
            cpp_storage_type = _hpro_cpp.storage_type_t.store_inverse
        else:
            raise RuntimeError('storage_type must be store_normal or store_inverse. storage_type=', me.storage_type)

        fac_options = _hpro_cpp.fac_options_t(cpp_eval_type, cpp_storage_type, me.coarsened)

        if me.factorization_type == 'LDL':
            LD_list = _hpro_cpp.split_ldl_factorization(me._factors_cpp_object, fac_options)

            L = HMatrix(LD_list[0], me.bct)
            D = HMatrix(LD_list[1], me.bct)
            return L, D
        elif me.factorization_type == 'LU':
            LU_list = _hpro_cpp.split_lu_factorization(me._factors_cpp_object, fac_options)

            L = HMatrix(LU_list[0], me.bct)
            U = HMatrix(LU_list[1], me.bct)
            return L, U
        else:
            raise RuntimeError('asdf')

def h_inv(A_hmatrix, rtol=default_rtol, atol=default_atol,
          overwrite=False, display_progress=True,
          diag_type='general_diag', storage_type='store_normal', do_coarsen=False):
    # Look in hpro/algebra/mat_inv.hh
    acc = _hpro_cpp.TTruncAcc(relative_eps=rtol, absolute_eps=atol)

    if do_coarsen:
        raise RuntimeError('coarsening inverse of hmatrix not implemented yet in python wrapper')

    if overwrite:
        inverse_cpp_object = A_hmatrix.cpp_object
    else:
        inverse_cpp_object = _hpro_cpp.copy_TMatrix(A_hmatrix.cpp_object)

    if diag_type == 'general_diag':
        cpp_diag_type = _hpro_cpp.diag_type_t.general_diag
    elif diag_type == 'unit_diag':
        cpp_diag_type = _hpro_cpp.diag_type_t.unit_diag
    else:
        raise RuntimeError('diag_type must be general_diag or unit_diag. diag_type=', diag_type)

    if storage_type == 'store_normal':
        cpp_storage_type = _hpro_cpp.storage_type_t.store_normal
    elif storage_type == 'store_inverse':
        cpp_storage_type = _hpro_cpp.storage_type_t.store_inverse
    else:
        raise RuntimeError('storage_type must be store_normal or store_inverse. storage_type=', storage_type)

    if display_progress:
        progress_bar = _hpro_cpp.TConsoleProgressBar()
        inv_options = _hpro_cpp.inv_options_t(cpp_diag_type, cpp_storage_type, do_coarsen, progress_bar)
    else:
        inv_options = _hpro_cpp.fac_options_t(cpp_diag_type, cpp_storage_type, do_coarsen)

    print('━━ H-matrix inverse ( rtol = ', rtol, ', atol = ', atol, ', overwrite=', overwrite, ' )')

    t = time()
    _hpro_cpp.invert_h_matrix(inverse_cpp_object, acc, inv_options)
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
    acc = _hpro_cpp.TTruncAcc(relative_eps=rtol, absolute_eps=atol)

    if overwrite:
        factors_cpp_object = A_hmatrix.cpp_object
    else:
        factors_cpp_object = _hpro_cpp.copy_TMatrix(A_hmatrix.cpp_object)

    if eval_type == 'point_wise':
        cpp_eval_type = _hpro_cpp.eval_type_t.point_wise
    elif eval_type == 'block_wise':
        cpp_eval_type = _hpro_cpp.eval_type_t.block_wise
    else:
        raise RuntimeError('eval_type must be point_wise or block_wise. eval_type=', eval_type)

    if storage_type == 'store_normal':
        cpp_storage_type = _hpro_cpp.storage_type_t.store_normal
    elif storage_type == 'store_inverse':
        cpp_storage_type = _hpro_cpp.storage_type_t.store_inverse
    else:
        raise RuntimeError('storage_type must be store_normal or store_inverse. storage_type=', storage_type)

    if display_progress:
        progress_bar = _hpro_cpp.TConsoleProgressBar()
        fac_options = _hpro_cpp.fac_options_t(cpp_eval_type, cpp_storage_type, do_coarsen, progress_bar)
    else:
        fac_options = _hpro_cpp.fac_options_t(cpp_eval_type, cpp_storage_type, do_coarsen)


    print('━━ ', operation_to_perform, ' factorisation ( rtol = ', rtol, ', atol = ', atol, ', overwrite=', overwrite, ' )')

    if operation_to_perform == 'LDL':
        t = time()
        _hpro_cpp.LDL_factorize(factors_cpp_object, acc, fac_options)
        dt_fac = time() - t
    elif operation_to_perform == 'LU':
        t = time()
        _hpro_cpp.LU_factorize(factors_cpp_object, acc, fac_options)
        dt_fac = time() - t

    print('    done in ', dt_fac)
    print('    size of factors = ', factors_cpp_object.byte_size(), ' bytes')

    eval_cpp_object = _hpro_cpp.LDL_eval_matrix(factors_cpp_object, fac_options)
    eval_inverse_cpp_object = _hpro_cpp.LDL_inv_matrix(factors_cpp_object, fac_options)

    return FactorizedHMatrix(eval_cpp_object, eval_inverse_cpp_object, factors_cpp_object, A_hmatrix.bct,
                             eval_type, storage_type, do_coarsen, operation_to_perform)




def build_hmatrix_from_scipy_sparse_matrix(A_csc, bct):
    A_csc[1,0] += 1e-14 # Force non-symmetry
    fname = "temp_sparse_matrix.mat"
    savemat(fname, {'A': A_csc})
    hmatrix_cpp_object = _hpro_cpp.build_hmatrix_from_sparse_matfile(fname, bct.cpp_object)
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
    PC_cpp = _hpro_cpp.ProductConvolution2d(WW_mins, WW_maxes, WW_arrays,
                                            FF_mins, FF_maxes, FF_arrays,
                                            row_dof_coords, col_dof_coords)

    PC_coefffn = _hpro_cpp.PC2DCoeffFn(PC_cpp)
    hmatrix_cpp_object = _hpro_cpp.build_hmatrix_from_coefffn(PC_coefffn, block_cluster_tree.cpp_object, tol)
    return HMatrix(hmatrix_cpp_object, block_cluster_tree)


def h_factorized_solve(iA_factorized, y):
    return _hpro_cpp.h_factorized_inverse_matvec(iA_factorized.cpp_object,
                                                 iA_factorized.row_ct.cpp_object,
                                                 iA_factorized.col_ct.cpp_object, y)


def h_matvec(A_hmatrix, x):
    return _hpro_cpp.h_matvec(A_hmatrix.cpp_object, A_hmatrix.row_ct.cpp_object, A_hmatrix.col_ct.cpp_object, x)

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
        _hpro_cpp.visualize_cluster_tree(me.cpp_object, filename)


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
        _hpro_cpp.visualize_block_cluster_tree(me.cpp_object, filename)


def build_block_cluster_tree(row_ct, col_ct, admissibility_eta=2.0):
    bct_cpp_object = _hpro_cpp.build_block_cluster_tree(row_ct.cpp_object, col_ct.cpp_object, admissibility_eta)
    return BlockClusterTree(bct_cpp_object, row_ct, col_ct)


def build_cluster_tree_from_pointcloud(points, cluster_size_cutoff=50):
    '''Build cluster tree from a collection of N points in d dimensions

    :param points: numpy array, shape=(N,d)
    :param cluster_size_cutoff: nonnegative int. number of points below which clusters are not subdivided further
    :return: BlockClusterTreeWrapper
    '''
    cpp_object = _hpro_cpp.build_cluster_tree_from_dof_coords(points, cluster_size_cutoff)
    return ClusterTree(cpp_object)

build_cluster_tree_from_dof_coords = build_cluster_tree_from_pointcloud




