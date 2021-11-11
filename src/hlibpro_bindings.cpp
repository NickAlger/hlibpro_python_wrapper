#include <cstdlib>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

#include <pybind11/pybind11.h>
#include <hlib.hh>
#include <hpro/algebra/mat_fac.hh>

#include <Eigen/Dense>
#include <Eigen/LU>
//#include <Eigen/CXX11/Tensor>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "grid_interpolate.h"
#include "product_convolution_hmatrix.h"
#include "rbf_interpolation.h"
#include "kdtree.h"
#include "aabbtree.h"
#include "simplexmesh.h"
#include "product_convolution_kernel.h"

using namespace Eigen;


// The order that the above two header files are loaded seems to affect the result slightly.

namespace py = pybind11;

using namespace std;
using namespace HLIB;

#if HLIB_SINGLE_PREC == 1
using  real_t = float;
#else
using  real_t = double;
#endif


std::shared_ptr<HLIB::TClusterTree> build_cluster_tree_from_dof_coords(const MatrixXd & dof_coords, const double nmin)
{
    size_t N = dof_coords.rows();
    size_t d = dof_coords.cols();

    vector< double * >  vertices( N );

    for ( size_t i = 0; i < N; i++ )
    {
        double * v    = new double[d];
        for (size_t j=0; j < d; ++j)
            v[j] = dof_coords(i,j);

        vertices[i] = v;
    }// for

    auto coord = make_unique< TCoordinate >( vertices, d );

//    TAutoBSPPartStrat  part_strat;
    TCardBSPPartStrat  part_strat;
    TBSPCTBuilder      ct_builder( & part_strat, nmin );
    std::unique_ptr<HLIB::TClusterTree>  ct = ct_builder.build( coord.get() );
//    return ct;
    return std::move(ct);
}

std::shared_ptr<HLIB::TBlockClusterTree> build_block_cluster_tree(std::shared_ptr<HLIB::TClusterTree> row_ct_ptr,
                                                                  std::shared_ptr<HLIB::TClusterTree> col_ct_ptr,
                                                                  double admissibility_eta)
{
        TStdGeomAdmCond    adm_cond( admissibility_eta );
        TBCBuilder         bct_builder;
        std::unique_ptr<HLIB::TBlockClusterTree>  bct = bct_builder.build( row_ct_ptr.get(),
                                                                           col_ct_ptr.get(), & adm_cond );
        return std::move(bct);
}

void initialize_hlibpro()
{
    int verbosity_level = 3;
    INIT();
    CFG::set_verbosity( verbosity_level );
}

void visualize_cluster_tree(std::shared_ptr<HLIB::TClusterTree> ct_ptr, string title)
{
    TPSClusterVis        c_vis;
    c_vis.print( ct_ptr.get()->root(), title );
}

void visualize_block_cluster_tree(std::shared_ptr<HLIB::TBlockClusterTree> bct_ptr, string title)
{
    TPSBlockClusterVis   bc_vis;
    bc_vis.print( bct_ptr.get()->root(), title );
}


std::shared_ptr<HLIB::TMatrix> build_hmatrix_from_sparse_matfile (string mat_file,
                                                                  std::shared_ptr<HLIB::TBlockClusterTree> bct_ptr)
{
    auto row_ct_ptr = bct_ptr.get()->row_ct();
    auto col_ct_ptr = bct_ptr.get()->col_ct();

    auto               M = read_matrix( mat_file );

    if ( ! IS_TYPE( M, TSparseMatrix ) )
    {
        cout << "given matrix is not sparse (" << M->typestr() << ")" << endl;
        exit( 1 );
    }

    auto               S = ptrcast( M.get(), TSparseMatrix );

    cout << "  matrix has dimension " << S->rows() << " x " << S->cols() << endl
         << "    no of non-zeroes    = " << S->n_non_zero() << endl
         << "    matrix is             " << ( S->is_complex() ? "complex" : "real" )
         << " valued" << endl
         << "    format              = ";
    if      ( S->is_nonsym()    ) cout << "non symmetric" << endl;
    else if ( S->is_symmetric() ) cout << "symmetric" << endl;
    else if ( S->is_hermitian() ) cout << "hermitian" << endl;
    cout << "  size of sparse matrix = " << Mem::to_string( S->byte_size() ) << endl;
    cout << "  |S|_F                 = " << norm_F( S ) << endl;

    cout << "    sparsity constant = " << bct_ptr.get()->compute_c_sp() << endl;

    TSparseMBuilder    h_builder( S, row_ct_ptr->perm_i2e(), col_ct_ptr->perm_e2i() );

    TTruncAcc                 acc(0.0, 0.0);
    auto               A = h_builder.build( bct_ptr.get(), acc );

    cout << "    size of H-matrix  = " << Mem::to_string( A->byte_size() ) << endl;
    cout << "    |A|_F             = " << norm_F( A.get() ) << endl;

    {
        auto  PA = make_unique< TPermMatrix >( row_ct_ptr->perm_i2e(), A.get(), col_ct_ptr->perm_e2i() );

        cout << " |S-A|_2 = " << diff_norm_2( S, PA.get() ) << endl;
    }

    return std::move(A);
//    return A;
}


std::shared_ptr<HLIB::TMatrix> build_hmatrix_from_coefffn(TCoeffFn<real_t> & coefffn,
                                                          std::shared_ptr<HLIB::TBlockClusterTree> bct_ptr,
                                                          double tol)
{
    const HLIB::TClusterTree * row_ct_ptr = bct_ptr.get()->row_ct();
    const HLIB::TClusterTree * col_ct_ptr = bct_ptr.get()->col_ct();
    std::cout << "━━ building H-matrix ( tol = " << tol << " )" << std::endl;
    TTimer                    timer( WALL_TIME );
    TConsoleProgressBar       progress;
    TTruncAcc                 acc( tol, 0.0 );
    TPermCoeffFn< real_t >    permuted_coefffn( & coefffn, row_ct_ptr->perm_i2e(), col_ct_ptr->perm_i2e() );
    TACAPlus< real_t >        aca( & permuted_coefffn );
    TDenseMBuilder< real_t >  h_builder( & permuted_coefffn, & aca );
    h_builder.set_coarsening( false );

    timer.start();

    std::unique_ptr<HLIB::TMatrix>  A = h_builder.build( bct_ptr.get(), acc, & progress );

    timer.pause();
    std::cout << "    done in " << timer << std::endl;
    std::cout << "    size of H-matrix = " << Mem::to_string( A->byte_size() ) << std::endl;
    return std::move(A);
}

void add_identity_to_hmatrix(HLIB::TMatrix * A_ptr, double s)
{
    add_identity(A_ptr, s);
}

void visualize_hmatrix(std::shared_ptr<HLIB::TMatrix> A_ptr, string title)
{
    TPSMatrixVis              mvis;
    mvis.svd( true );
    mvis.print( A_ptr.get(), title );
}

VectorXd h_matvec(std::shared_ptr<HLIB::TMatrix> A_ptr,
                  std::shared_ptr<HLIB::TClusterTree> row_ct_ptr,
                  std::shared_ptr<HLIB::TClusterTree> col_ct_ptr,
                  VectorXd x)
{
    // y = A * x
    std::unique_ptr<HLIB::TVector> y_hlib = A_ptr.get()->row_vector();
    std::unique_ptr<HLIB::TVector> x_hlib = A_ptr.get()->col_vector();

    int n = y_hlib->size();
    int m = x_hlib->size();
//    int n = x.size();
//    int m = x.size();

    for ( size_t  i = 0; i < m; i++ )
        x_hlib->set_entry( i, x(i) );

//    col_ct_ptr->perm_e2i()->permute( x_hlib.get() );
    col_ct_ptr.get()->perm_e2i()->permute( x_hlib.get() );

    A_ptr->apply(x_hlib.get(), y_hlib.get());

//    row_ct_ptr->perm_i2e()->permute( y_hlib.get() );
    row_ct_ptr.get()->perm_i2e()->permute( y_hlib.get() );

    VectorXd y(n);
    for ( size_t  i = 0; i < n; i++ )
        y(i) = y_hlib->entry( i );

    return y;
}

VectorXd h_factorized_inverse_matvec(HLIB::TFacInvMatrix * inv_A_ptr,
                                     HLIB::TClusterTree * row_ct_ptr,
                                     HLIB::TClusterTree * col_ct_ptr,
                                     VectorXd x)
{
    // y = inv_A * x
    std::unique_ptr<HLIB::TVector> y_hlib = inv_A_ptr->matrix()->row_vector();
    std::unique_ptr<HLIB::TVector> x_hlib = inv_A_ptr->matrix()->col_vector();

//    int n = y_hlib->size();
//    int m = x_hlib->size();
    int n = x.size();
    int m = x.size();

    for ( size_t  i = 0; i < m; i++ )
        x_hlib->set_entry( i, x(i) );

    col_ct_ptr->perm_e2i()->permute( x_hlib.get() );

    inv_A_ptr->apply(x_hlib.get(), y_hlib.get());

    row_ct_ptr->perm_i2e()->permute( y_hlib.get() );

    VectorXd y(n);
    for ( size_t  i = 0; i < n; i++ )
        y(i) = y_hlib->entry( i );

    return y;
}

VectorXd h_factorized_matvec(HLIB::TFacMatrix * A_factorized_ptr,
                             HLIB::TClusterTree * row_ct_ptr,
                             HLIB::TClusterTree * col_ct_ptr,
                             VectorXd x)
{
    // y = A * x
    std::unique_ptr<HLIB::TVector> y_hlib = A_factorized_ptr->matrix()->row_vector();
    std::unique_ptr<HLIB::TVector> x_hlib = A_factorized_ptr->matrix()->col_vector();

    int n = x.size();
    int m = x.size();

    for ( size_t  i = 0; i < m; i++ )
        x_hlib->set_entry( i, x(i) );

    col_ct_ptr->perm_e2i()->permute( x_hlib.get() );

    A_factorized_ptr->apply(x_hlib.get(), y_hlib.get());

    row_ct_ptr->perm_i2e()->permute( y_hlib.get() );

    VectorXd y(n);
    for ( size_t  i = 0; i < n; i++ )
        y(i) = y_hlib->entry( i );

    return y;
}

//std::unique_ptr<HLIB::TFacInvMatrix> factorize_inv_with_progress_bar(HLIB::TMatrix * A_ptr, TTruncAcc acc)
std::shared_ptr<HLIB::TFacInvMatrix> factorize_inv_with_progress_bar(HLIB::TMatrix * A_ptr, TTruncAcc acc)
{
    double rtol = acc.rel_eps();
    TTimer                    timer( WALL_TIME );
    TConsoleProgressBar       progress;

    std::cout << std::endl << "━━ LU factorisation ( rtol = " << rtol << " )" << std::endl;

    timer.start();

    std::unique_ptr<HLIB::TFacInvMatrix> A_inv = factorise_inv( A_ptr, acc, & progress );

    timer.pause();
    std::cout << "    done in " << timer << std::endl;

    std::cout << "    size of LU factor = " << Mem::to_string( A_ptr->byte_size() ) << std::endl;

//    return A_inv;
    return std::move(A_inv);
}

std::tuple<std::shared_ptr<HLIB::TFacMatrix>, std::shared_ptr<HLIB::TFacInvMatrix>> ldl_factorization_inplace(HLIB::TMatrix * A_ptr, TTruncAcc acc)
{
    double rtol = acc.rel_eps();
    TTimer                    timer( WALL_TIME );
    TConsoleProgressBar       progress;

    fac_options_t  facopt(point_wise, store_normal, false, & progress);
//    LDL::TLDL        fac(facopt);

    std::cout << std::endl << "━━ LDLt factorisation ( rtol = " << rtol << " )" << std::endl;

    timer.start();

//    fac.factorise( A_ptr, acc );
    LDL::factorise( A_ptr, acc, facopt );

    timer.pause();
    std::cout << "    done in " << timer << std::endl;

    std::cout << "    size of LDLt factor = " << Mem::to_string( A_ptr->byte_size() ) << std::endl;

//    std::unique_ptr<TLDLMatrix> A_fac = fac.eval_matrix( A_ptr );
    std::shared_ptr<HLIB::TFacMatrix> A_fac = std::move(LDL::eval_matrix( A_ptr, A_ptr->form(), facopt ));
    std::shared_ptr<HLIB::TFacInvMatrix> A_invfac = std::move(LDL::inv_matrix( A_ptr, A_ptr->form(), facopt ));
    return std::make_tuple(A_fac, A_invfac);
}

//std::unique_ptr< HLIB::TFacInvMatrix > hmatrix_factorized_inverse_destructive(HLIB::TMatrix * A_ptr, double tol)
std::shared_ptr< HLIB::TFacInvMatrix > hmatrix_factorized_inverse_destructive(HLIB::TMatrix * A_ptr, double tol)
{
    TTruncAcc                 acc( tol, 0.0 );
    TTimer                    timer( WALL_TIME );
    TConsoleProgressBar       progress;

    std::cout << std::endl << "━━ LU factorisation ( tol = " << tol << " )" << std::endl;

    timer.start();

    std::unique_ptr<HLIB::TFacInvMatrix> A_inv = factorise_inv( A_ptr, acc, & progress );

    timer.pause();
    std::cout << "    done in " << timer << std::endl;

    std::cout << "    size of LU factor = " << Mem::to_string( A_ptr->byte_size() ) << std::endl;

//    return A_inv;
    return std::move(A_inv);
}


void hmatrix_add_overwrites_second (const HLIB::TMatrix * A, HLIB::TMatrix* B, double tol)
{
    TTruncAcc                 acc( tol, 0.0 );
    add(1.0, A, 1.0, B, acc);
}

template < typename T_value >
void multiply_without_progress_bar(const T_value  	    alpha,
		                           const matop_t  	    op_A,
                                   const TMatrix *      A,
                                   const matop_t  	    op_B,
                                   const TMatrix *      B,
                                   const T_value  	    beta,
                                   TMatrix *  	        C,
                                   const TTruncAcc &  	acc)
{
    multiply(alpha, op_A, A, op_B, B, beta, C, acc);
}

template < typename T_value >
void multiply_with_progress_bar(const T_value  	    alpha,
		                        const matop_t  	    op_A,
                                const TMatrix *      A,
                                const matop_t  	    op_B,
                                const TMatrix *      B,
                                const T_value  	    beta,
                                TMatrix *  	        C,
                                const TTruncAcc &  	acc)
{
//    TTruncAcc              acc2( 1e-6 );
    TTimer                    timer( WALL_TIME );
    TConsoleProgressBar       progress;

    std::cout << std::endl << "━━ H-matrix multiplication C=A*B " << std::endl;

    timer.start();

    multiply(alpha, op_A, A, op_B, B, beta, C, acc, & progress);

    timer.pause();
    std::cout << "    done in " << timer << std::endl;

    std::cout << "    size of C = " << Mem::to_string( C->byte_size() ) << std::endl;
}


std::shared_ptr<HLIB::TMatrix> copy_TMatrix(std::shared_ptr<HLIB::TMatrix> A)
{
    std::unique_ptr<HLIB::TMatrix> A_copy = A->copy();
    return std::move(A_copy);
}

void copy_TMatrix_into_another_TMatrix(HLIB::TMatrix * source, HLIB::TMatrix * target)
{
    source->copy_to(target);
}

std::shared_ptr<HLIB::TMatrix> copy_struct_TMatrix(std::shared_ptr<HLIB::TMatrix> A)
{
    std::unique_ptr<HLIB::TMatrix> A_copy_struct = A.get()->copy_struct();
    return std::move(A_copy_struct);
}


std::shared_ptr<HLIB::TFacMatrix> LDL_eval_matrix(std::shared_ptr<HLIB::TMatrix> A, fac_options_t facopt)
{
    return std::move(LDL::eval_matrix( A.get(), A->form(), facopt ));
}

std::shared_ptr<HLIB::TFacInvMatrix> LDL_inv_matrix(std::shared_ptr<HLIB::TMatrix> A, fac_options_t facopt)
{
    return std::move(LDL::inv_matrix( A.get(), A->form(), facopt ));
}

std::shared_ptr<HLIB::TFacMatrix> LU_eval_matrix(std::shared_ptr<HLIB::TMatrix> A, fac_options_t facopt)
{
    return std::move(LU::eval_matrix( A.get(), facopt ));
}

std::shared_ptr<HLIB::TFacInvMatrix> LU_inv_matrix(std::shared_ptr<HLIB::TMatrix> A, fac_options_t facopt)
{
    return std::move(LU::inv_matrix( A.get(), facopt ));
}


std::vector<std::shared_ptr<HLIB::TMatrix>>
    split_ldl_factorization(std::shared_ptr<HLIB::TMatrix> A, const fac_options_t opts)
{
    std::pair<HLIB::TMatrix*, HLIB::TMatrix*> LD_pair = LDL::split(A.get(), opts);
    std::shared_ptr<HLIB::TMatrix> L(LD_pair.first->copy());
    std::shared_ptr<HLIB::TMatrix> D(LD_pair.second->copy());

//    cout << "L byte size" << L.get()->byte_size() << endl;

//    return L;
    std::vector<std::shared_ptr<HLIB::TMatrix>> LD_list(2);
    LD_list[0] = L;
    LD_list[1] = D;
    return LD_list;
}


std::vector<std::shared_ptr<HLIB::TMatrix>>
    split_lu_factorization(std::shared_ptr<HLIB::TMatrix> A, const fac_options_t opts)
{
    std::pair<HLIB::TMatrix*, HLIB::TMatrix*> LU_pair = LU::split(A.get(), opts);
    std::shared_ptr<HLIB::TMatrix> L(LU_pair.first->copy());
    std::shared_ptr<HLIB::TMatrix> U(LU_pair.second->copy());

    std::vector<std::shared_ptr<HLIB::TMatrix>> LU_list(2);
    LU_list[0] = L;
    LU_list[1] = U;
    return LU_list;
}

void print_hello() {
    std::cout << "hello" << std::endl;
}


PYBIND11_MODULE(hlibpro_bindings, m) {
    m.doc() = "hlibpro wrapper plus product convolution hmatrix stuff";

    // -----------------------------------------------
    // --------      H-matrix Bindings        --------
    // -----------------------------------------------

    m.def("print_hello", &print_hello);


    py::class_<HLIB::TProgressBar>(m, "TProgressBar");
//        .def(py::init<>());
//        .def(py::init<const double, const double, const double>(), py::arg("amin"), py::arg("amax"), py::arg("acur"));

    py::class_<HLIB::TConsoleProgressBar, HLIB::TProgressBar>(m, "TConsoleProgressBar")
        .def(py::init<>());

    py::class_<HLIB::TFacMatrix, std::shared_ptr<TFacMatrix>>(m, "TFacMatrix");
    py::class_<HLIB::TFacInvMatrix, std::shared_ptr<TFacInvMatrix>>(m, "TFacInvMatrix");
    py::class_<HLIB::TBlockCluster>(m, "TBlockCluster");

//    py::class_<HLIB::TLDLMatrix, std::shared_ptr<HLIB::TLDLMatrix>>(m, "TLDLMatrix");

    py::class_<HLIB::TBlockMatrix>(m, "TBlockMatrix")
        .def(py::init<const TBlockCluster *>(), py::arg("bct")=nullptr);

    py::enum_<HLIB::matop_t>(m, "matop_t")
        .value("MATOP_NORM", HLIB::matop_t::MATOP_NORM)
        .value("apply_normal", HLIB::matop_t::apply_normal)
        .value("MATOP_TRANS", HLIB::matop_t::MATOP_TRANS)
        .value("apply_trans", HLIB::matop_t::apply_trans)
        .value("apply_transposed", HLIB::matop_t::apply_transposed)
        .value("MATOP_ADJ", HLIB::matop_t::MATOP_ADJ)
        .value("MATOP_CONJTRANS", HLIB::matop_t::MATOP_CONJTRANS)
        .value("apply_adj", HLIB::matop_t::apply_adj)
        .value("apply_adjoint", HLIB::matop_t::apply_adjoint)
        .value("apply_conjtrans", HLIB::matop_t::apply_conjtrans)
        .export_values();

    py::class_<HLIB::TTruncAcc>(m, "TTruncAcc")
        .def("max_rank", &HLIB::TTruncAcc::max_rank)
        .def("has_max_rank", &HLIB::TTruncAcc::has_max_rank)
        .def("rel_eps", &HLIB::TTruncAcc::rel_eps)
        .def("abs_eps", &HLIB::TTruncAcc::abs_eps)
        .def(py::init<>())
//        .def(py::init<const int, double>(), py::arg("k"), py::arg("absolute_eps")=CFG::Arith::abs_eps)
        .def(py::init<const double, double>(), py::arg("relative_eps"), py::arg("absolute_eps")=CFG::Arith::abs_eps);

    py::class_<HLIB::TVector>(m, "TVector");

    py::class_<HLIB::TMatrix, std::shared_ptr<HLIB::TMatrix>>(m, "TMatrix")
//    py::class_<HLIB::TMatrix, std::unique_ptr<HLIB::TMatrix>>(m, "TMatrix")
        .def("id", &HLIB::TMatrix::id)
        .def("rows", &HLIB::TMatrix::rows)
        .def("cols", &HLIB::TMatrix::cols)
        .def("is_nonsym", &HLIB::TMatrix::is_nonsym)
        .def("is_symmetric", &HLIB::TMatrix::is_symmetric)
        .def("is_hermitian", &HLIB::TMatrix::is_hermitian)
        .def("set_nonsym", &HLIB::TMatrix::set_nonsym)
        .def("is_real", &HLIB::TMatrix::is_real)
        .def("is_complex", &HLIB::TMatrix::is_complex)
        .def("to_real", &HLIB::TMatrix::to_real)
        .def("to_complex", &HLIB::TMatrix::to_complex)
        .def("add_update", &HLIB::TMatrix::add_update)
        .def("entry", &HLIB::TMatrix::entry)
        .def("apply", &HLIB::TMatrix::apply, py::arg("x"), py::arg("y"), py::arg("op")=apply_normal)
//        .def("apply_add", static_cast<void
//                                      (HLIB::TMatrix::*)(const real_t,
//                                                         const HLIB::TVector *,
//                                                         HLIB::TVector *,
//                                                         const matop_t) const>(&HLIB::TMatrix::apply_add),
//                                      py::arg("alpha"), py::arg("x"), py::arg("y"), py::arg("op")=apply_normal)
        .def("set_symmetric", &HLIB::TMatrix::set_symmetric)
        .def("set_hermitian", &HLIB::TMatrix::set_hermitian)
        .def("domain_dim", &HLIB::TMatrix::domain_dim)
        .def("range_dim", &HLIB::TMatrix::range_dim)
        .def("domain_vector", &HLIB::TMatrix::domain_vector)
        .def("range_vector", &HLIB::TMatrix::range_vector)
        .def("transpose", &HLIB::TMatrix::transpose)
        .def("conjugate", &HLIB::TMatrix::conjugate)
        .def("add", &HLIB::TMatrix::add)
        .def("scale", &HLIB::TMatrix::scale)
        .def("truncate", &HLIB::TMatrix::truncate)
        .def("mul_vec", &HLIB::TMatrix::mul_vec, py::arg("alpha"), py::arg("x"),
                                                 py::arg("beta"), py::arg("y"), py::arg("op")=MATOP_NORM)
//        .def("mul_right", &HLIB::TMatrix::mul_right) // not implemented apparently
//        .def("mul_left", &HLIB::TMatrix::mul_left) // not implemented apparently
        .def("check_data", &HLIB::TMatrix::check_data)
        .def("byte_size", &HLIB::TMatrix::byte_size)
        .def("print", &HLIB::TMatrix::print)
//        .def("copy", static_cast<std::unique_ptr<TMatrix> (HLIB::TMatrix::*)() const>(&HLIB::TMatrix::copy))
//        .def("copy", static_cast<std::shared_ptr<TMatrix> (HLIB::TMatrix::*)() const>(&HLIB::TMatrix::copy))
//        .def("copy", &HLIB::TMatrix::copy)
//        .def("copy_struct", &HLIB::TMatrix::copy_struct)
        .def("create", &HLIB::TMatrix::create)
        .def("cluster", &HLIB::TMatrix::cluster)
        .def("row_vector", &HLIB::TMatrix::row_vector)
        .def("col_vector", &HLIB::TMatrix::col_vector)
        .def("form", &HLIB::TMatrix::form);

    m.def("copy_TMatrix", &copy_TMatrix);
    m.def("copy_struct_TMatrix", &copy_struct_TMatrix);
    m.def("copy_TMatrix_into_another_TMatrix", &copy_TMatrix_into_another_TMatrix);

    m.def("add", &add<real_t>);
    m.def("multiply_without_progress_bar", &multiply_without_progress_bar<real_t>);
    m.def("multiply_with_progress_bar", &multiply_with_progress_bar<real_t>);
    m.def("factorize_inv_with_progress_bar", &factorize_inv_with_progress_bar);

    py::class_<HLIB::TCoeffFn<real_t>>(m, "TCoeffFn<real_t>");

//    py::class_<HLIB::TClusterTree>(m, "HLIB::TClusterTree")
    py::class_<HLIB::TClusterTree, std::shared_ptr<TClusterTree>>(m, "HLIB::TClusterTree")
        .def("perm_i2e", &HLIB::TClusterTree::perm_i2e)
        .def("perm_e2i", &HLIB::TClusterTree::perm_e2i)
        .def("nnodes", &HLIB::TClusterTree::nnodes)
        .def("depth", &HLIB::TClusterTree::depth)
        .def("byte_size", &HLIB::TClusterTree::byte_size);

    py::class_<HLIB::TBlockClusterTree, std::shared_ptr<HLIB::TBlockClusterTree>>(m, "HLIB::TBlockClusterTree")
//    py::class_<HLIB::TBlockClusterTree>(m, "HLIB::TBlockClusterTree")
        .def("row_ct", &HLIB::TBlockClusterTree::row_ct)
        .def("col_ct", &HLIB::TBlockClusterTree::col_ct)
        .def("nnodes", &HLIB::TBlockClusterTree::nnodes)
        .def("depth", &HLIB::TBlockClusterTree::depth)
        .def("nnodes", &HLIB::TBlockClusterTree::nnodes)
        .def("depth", &HLIB::TBlockClusterTree::depth)
        .def("compute_c_sp", &HLIB::TBlockClusterTree::compute_c_sp)
        .def("byte_size", &HLIB::TBlockClusterTree::byte_size);

    m.def("build_cluster_tree_from_dof_coords", &build_cluster_tree_from_dof_coords);
    m.def("build_block_cluster_tree", &build_block_cluster_tree);
    m.def("initialize_hlibpro", &initialize_hlibpro);
    m.def("visualize_cluster_tree", &visualize_cluster_tree);
    m.def("visualize_block_cluster_tree", &visualize_block_cluster_tree);
    m.def("build_hmatrix_from_coefffn", &build_hmatrix_from_coefffn);
    m.def("add_identity_to_hmatrix", &add_identity_to_hmatrix);
    m.def("visualize_hmatrix", &visualize_hmatrix);
    m.def("h_matvec", &h_matvec);
    m.def("hmatrix_factorized_inverse_destructive", &hmatrix_factorized_inverse_destructive);
    m.def("h_factorized_inverse_matvec", &h_factorized_inverse_matvec);
    m.def("build_hmatrix_from_sparse_matfile", &build_hmatrix_from_sparse_matfile);
    m.def("hmatrix_add_overwrites_second", &hmatrix_add_overwrites_second);

    // ----------------------------------------------------------
    // --------      Product Convolution Bindings        --------
    // ----------------------------------------------------------

    py::class_<ProductConvolutionCoeffFn, HLIB::TCoeffFn<real_t>>(m, "ProductConvolutionCoeffFn")
        .def(py::init<const ProductConvolutionMultipleBatches &, MatrixXd>());

    py::class_<ProductConvolutionOneBatch>(m, "ProductConvolutionOneBatch")
        .def(py::init<MatrixXd, // eta
             std::vector<MatrixXd>, // ww
             MatrixXd, // pp
             MatrixXd, // mus
             std::vector<MatrixXd>, // Sigmas
             double, // tau
             double, // xmin
             double, // xmax
             double, // ymin
             double  // ymax
             >())
        .def("compute_entries", &ProductConvolutionOneBatch::compute_entries);

    py::class_<ProductConvolutionMultipleBatches>(m, "ProductConvolutionMultipleBatches")
        .def(py::init<std::vector<MatrixXd>, // eta_array_batches
                      std::vector<std::vector<MatrixXd>>, // ww_array_batches
                      std::vector<MatrixXd>, // pp_batches
                      std::vector<MatrixXd>, // mus_batches
                      std::vector<std::vector<MatrixXd>>, // Sigmas_batches
                      double, // tau
                      double, // xmin
                      double, // xmax
                      double, // ymin
                      double // ymax
                      >())
        .def("compute_entries", &ProductConvolutionMultipleBatches::compute_entries);

    m.def("grid_interpolate", &grid_interpolate);
    m.def("grid_interpolate_vectorized", &grid_interpolate_vectorized);
    m.def("point_is_in_ellipsoid", &point_is_in_ellipsoid);

    m.def("periodic_bilinear_interpolation_regular_grid", &periodic_bilinear_interpolation_regular_grid);
    m.def("bilinear_interpolation_regular_grid", &bilinear_interpolation_regular_grid);

    py::class_<ProductConvolution2d>(m, "ProductConvolution2d")
        .def(py::init<std::vector<Vector2d>, // WW_mins
                      std::vector<Vector2d>, // WW_maxes
                      std::vector<MatrixXd>, // WW_arrays
                      std::vector<Vector2d>, // FF_mins
                      std::vector<Vector2d>, // FF_maxes
                      std::vector<MatrixXd>, // FF_arrays
                      Matrix<double, Dynamic, 2>, // row_coords
                      Matrix<double, Dynamic, 2>  // col_coords
                      >())
        .def("get_entries", &ProductConvolution2d::get_entries)
        .def("get_block", &ProductConvolution2d::get_block)
        .def("get_array", &ProductConvolution2d::get_array);

    py::class_<PC2DCoeffFn, HLIB::TCoeffFn<real_t>>(m, "PC2DCoeffFn")
        .def(py::init<const ProductConvolution2d &>());


    //

    m.def("ldl_factorization_inplace", &ldl_factorization_inplace);
    m.def("h_factorized_matvec", &h_factorized_matvec);

    py::enum_<HLIB::eval_type_t>(m, "eval_type_t", py::arithmetic())
        .value("point_wise", HLIB::eval_type_t::point_wise)
        .value("block_wise", HLIB::eval_type_t::block_wise)
        .export_values();

    py::enum_<HLIB::storage_type_t>(m, "storage_type_t", py::arithmetic())
        .value("store_normal", HLIB::storage_type_t::store_normal)
        .value("store_inverse", HLIB::storage_type_t::store_inverse)
        .export_values();

//    py::class_<HLIB::fac_options_t, std::shared_ptr<HLIB::fac_options_t>>(m, "fac_options_t")
    py::class_<HLIB::fac_options_t, std::shared_ptr<HLIB::fac_options_t>>(m, "fac_options_t")
        .def(py::init<>())
        .def(py::init<const eval_type_t, const storage_type_t, const bool, TProgressBar *>(),
             py::arg("aeval"),
             py::arg("astorage"),
             py::arg("ado_coarsen"),
             py::arg("aprogress")=nullptr);

    m.def("LDL_factorize", &LDL::factorise);
    m.def("LDL_eval_matrix", &LDL_eval_matrix);
    m.def("LDL_inv_matrix", &LDL_inv_matrix);

    m.def("LU_factorize", &LU::factorise);
    m.def("LU_eval_matrix", &LU_eval_matrix);
    m.def("LU_inv_matrix", &LU_inv_matrix);

    py::enum_<HLIB::matform_t>(m, "matform_t", py::arithmetic())
        .value("unsymmetric", HLIB::matform_t::unsymmetric)
        .value("symmetric", HLIB::matform_t::symmetric)
        .value("hermitian", HLIB::matform_t::hermitian)
        .value("MATFORM_NONSYM", HLIB::matform_t::MATFORM_NONSYM)
        .value("MATFORM_SYM", HLIB::matform_t::MATFORM_SYM)
        .value("MATFORM_HERM", HLIB::matform_t::MATFORM_HERM)
        .export_values();

    m.def("split_ldl_factorization", &split_ldl_factorization);
    m.def("split_lu_factorization", &split_lu_factorization);

    py::enum_<HLIB::BLAS::diag_type_t>(m, "diag_type_t", py::arithmetic())
        .value("unit_diag", HLIB::BLAS::diag_type_t::unit_diag)
        .value("general_diag", HLIB::BLAS::diag_type_t::general_diag)
        .export_values();

    py::class_<HLIB::inv_options_t, std::shared_ptr<HLIB::inv_options_t>>(m, "inv_options_t")
        .def(py::init<>())
        .def(py::init<const diag_type_t, const storage_type_t, const bool, TProgressBar *>(),
             py::arg("adiag")=general_diag,
             py::arg("astorage")=store_normal,
             py::arg("acoarsen")=CFG::Arith::coarsen,
             py::arg("aprogress")=nullptr);

    m.def("invert_h_matrix", static_cast<void (*)(TMatrix *, const TTruncAcc &, const inv_options_t &)>(&HLIB::invert));

    m.def("eval_thin_plate_splines_at_points", &eval_thin_plate_splines_at_points);

    py::class_<ThinPlateSplineWeightingFunctions>(m, "ThinPlateSplineWeightingFunctions")
        .def(py::init< Array<double, Dynamic, 2> >())
        .def("eval_weighting_functions", &ThinPlateSplineWeightingFunctions::eval_weighting_functions);


    py::class_<KDTree<2>>(m, "KDTree2D")
        .def(py::init< const vector<Matrix<double,2,1>> & >())
        .def("nearest_neighbor", py::overload_cast< const Matrix<double,2,1> &                 >(&KDTree<2>::nearest_neighbor, py::const_), "one query, one neighbor")
        .def("nearest_neighbor", py::overload_cast< const Matrix<double,2,1> &, int >(&KDTree<2>::nearest_neighbor, py::const_), "one query, many neighbors")
        .def("nearest_neighbor_vectorized", py::overload_cast< const Ref<const Matrix<double,2,Dynamic>>      >(&KDTree<2>::nearest_neighbor_vectorized, py::const_), "many querys, one neighbor")
        .def("nearest_neighbor_vectorized", py::overload_cast< const Ref<const Matrix<double,2,Dynamic>>, int >(&KDTree<2>::nearest_neighbor_vectorized, py::const_), "many querys, many neighbors");


    py::class_<AABBTree<2>>(m, "AABBTree2D")
        .def(py::init< const Ref<const Array<double, 2, Dynamic>>,
                       const Ref<const Array<double, 2, Dynamic>> >())
        .def("point_collisions", &AABBTree<2>::point_collisions)
        .def("point_collisions_vectorized", &AABBTree<2>::point_collisions_vectorized)
        .def("ball_collisions", &AABBTree<2>::ball_collisions)
        .def("ball_collisions_vectorized", &AABBTree<2>::ball_collisions_vectorized);

    py::class_<SimplexMesh<2>>(m, "SimplexMesh2D")
        .def(py::init< const Ref<const Array<double, 2, Dynamic>>,
                       const Ref<const Array<int, 3, Dynamic>> >())
        .def("closest_point", &SimplexMesh<2>::closest_point)
        .def("closest_point_vectorized", &SimplexMesh<2>::closest_point_vectorized)
//        .def("closest_point_vectorized_multithreaded", &SimplexMesh<2>::closest_point_vectorized_multithreaded)
        .def("point_is_in_mesh", &SimplexMesh<2>::point_is_in_mesh)
        .def("point_is_in_mesh_vectorized", &SimplexMesh<2>::point_is_in_mesh_vectorized)
        .def("index_of_first_simplex_containing_point", &SimplexMesh<2>::index_of_first_simplex_containing_point)
        .def("evaluate_functions_at_points", &SimplexMesh<2>::evaluate_functions_at_points)
        .def("set_sleep_duration", &SimplexMesh<2>::set_sleep_duration)
        .def("reset_sleep_duration_to_default", &SimplexMesh<2>::reset_sleep_duration_to_default)
        .def("set_thread_count", &SimplexMesh<2>::set_thread_count)
        .def("reset_thread_count_to_default", &SimplexMesh<2>::reset_thread_count_to_default)
        .def("evaluate_functions_at_points_with_reflection", &SimplexMesh<2>::evaluate_functions_at_points_with_reflection)
        .def("evaluate_functions_at_points_with_reflection_and_ellipsoid_truncation", &SimplexMesh<2>::evaluate_functions_at_points_with_reflection_and_ellipsoid_truncation);

    m.def("projected_affine_coordinates", &projected_affine_coordinates);
    m.def("closest_point_in_simplex_vectorized", &closest_point_in_simplex_vectorized);
    m.def("powerset", &powerset);
    m.def("submatrix_deletion_factors", &submatrix_deletion_factors);
//    m.def("woodbury_update", &woodbury_update);


    py::class_<ProductConvolutionKernelRBF<2>, HLIB::TCoeffFn<real_t>>(m, "ProductConvolutionKernelRBF")
        .def(py::init< shared_ptr<ImpulseResponseBatches<2>>, // IRO_FWD,
                       shared_ptr<ImpulseResponseBatches<2>>, // IRO_ADJ,
                       vector<Matrix<double, 2, 1>>,          // row_coords,
                       vector<Matrix<double, 2, 1>>           // col_coords
                       >())
        .def("eval_integral_kernel", &ProductConvolutionKernelRBF<2>::eval_integral_kernel)
        .def("eval_integral_kernel_block", &ProductConvolutionKernelRBF<2>::eval_integral_kernel_block);

    m.def("tps_interpolate_vectorized", &tps_interpolate_vectorized);
    m.def("nearest_points_brute_force_vectorized", &nearest_points_brute_force_vectorized);

    py::class_<ImpulseResponseBatches<2>, shared_ptr<ImpulseResponseBatches<2>>>(m, "ImpulseResponseBatches")
        .def(py::init< const Ref<const Matrix<double, 2, Dynamic>>, // mesh_vertices,
                       const Ref<const Matrix<int   , 3, Dynamic>>, // mesh_cells,
                       int,                                         // num_neighbors,
                       double                                       // tau
                       >())
        .def_readwrite("tau", &ImpulseResponseBatches<2>::tau)
        .def_readwrite("num_neighbors", &ImpulseResponseBatches<2>::num_neighbors)
        .def_readonly("kdtree", &ImpulseResponseBatches<2>::kdtree)
        .def_readonly("mesh", &ImpulseResponseBatches<2>::mesh)
        .def_readonly("pts", &ImpulseResponseBatches<2>::pts)
        .def_readonly("mu", &ImpulseResponseBatches<2>::mu)
        .def_readonly("inv_Sigma", &ImpulseResponseBatches<2>::inv_Sigma)
        .def_readonly("psi_batches", &ImpulseResponseBatches<2>::psi_batches)
        .def_readonly("point2batch", &ImpulseResponseBatches<2>::point2batch)
        .def_readonly("batch2point_start", &ImpulseResponseBatches<2>::batch2point_start)
        .def_readonly("batch2point_stop", &ImpulseResponseBatches<2>::batch2point_stop)
        .def("num_pts", &ImpulseResponseBatches<2>::num_pts)
        .def("num_batches", &ImpulseResponseBatches<2>::num_batches)
        .def("add_batch", &ImpulseResponseBatches<2>::add_batch)
        .def("build_kdtree", &ImpulseResponseBatches<2>::build_kdtree)
        .def("interpolation_points_and_values", &ImpulseResponseBatches<2>::interpolation_points_and_values);
}

