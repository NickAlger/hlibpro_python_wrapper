#include <cstdlib>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

#include "grid_interpolate.h"
#include "product_convolution_hmatrix.h"

#include <pybind11/pybind11.h>
#include <hlib.hh>

#include <Eigen/Dense>
#include <Eigen/LU>
//#include <Eigen/CXX11/Tensor>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

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


bool point_is_in_ellipsoid(VectorXd z, VectorXd mu, MatrixXd Sigma, double tau)
{
    VectorXd p = z - mu;
    return ( p.dot(Sigma.lu().solve(p)) < pow(tau, 2) );
}


// --------------------------------------------------------
// -----------      ProductConvolution2d        -----------
// --------------------------------------------------------


ProductConvolution2d::ProductConvolution2d(Matrix<double, Dynamic, 2> WW_mins,
                                           Matrix<double, Dynamic, 2> WW_maxes,
                                           std::vector<MatrixXd>      WW_arrays,
                                           Matrix<double, Dynamic, 2> FF_mins,
                                           Matrix<double, Dynamic, 2> FF_maxes,
                                           std::vector<MatrixXd>      FF_arrays,
                                           Matrix<double, Dynamic, 2> row_coords,
                                           Matrix<double, Dynamic, 2> col_coords) :
                                           WW_mins(WW_mins),
                                           WW_maxes(WW_maxes),
                                           WW_arrays(WW_arrays),
                                           FF_mins(FF_mins),
                                           FF_maxes(FF_maxes),
                                           FF_arrays(FF_arrays),
                                           row_coords(row_coords),
                                           col_coords(col_coords),
                                           num_patches(WW_mins.rows()),
                                           num_rows(row_coords.rows()),
                                           num_cols(col_coords.rows()),
                                           row_patches(row_coords.rows()),
                                           col_patches(col_coords.rows())
{
    // Compute patches relevant to each row
    for (int r=0; r < num_rows; ++r)
    {
        const Vector2d x = row_coords.row(r);
        for (int p=0; p < num_patches; ++p)
        {
            const Vector2d x_min = WW_mins.row(p);
            const Vector2d x_max = WW_maxes.row(p);
            if (x_min(0) <= x(0) && x(0) <= x_max(0) &&
                x_min(1) <= x(1) && x(1) <= x_max(1))
            {
                row_patches[r].push_back(p);
            }
        }
    }

    // Compute patches relevant to each col
    for (int c=0; c < num_cols; ++c)
    {
        const Vector2d y = col_coords.row(c);
        for (int p=0; p < num_patches; ++p)
        {
            const Vector2d y_min = WW_mins.row(p) + FF_mins.row(p);
            const Vector2d y_max = WW_maxes.row(p) + FF_mins.row(p);
            if (y_min(0) <= y(0) && y(0) <= y_max(0) &&
                y_min(1) <= y(1) && y(1) <= y_max(1))
            {
                col_patches[c].push_back(p);
            }
        }
    }
}

//ProductConvolution2d::get_entries(VectorXi rows, VectorXi cols)
//{
//
//}

// --------------------------------------------------------
// --------      ProductConvolutionOneBatch        --------
// --------------------------------------------------------

ProductConvolutionOneBatch::ProductConvolutionOneBatch()
    : xmin(), xmax(), ymin(), ymax(),
      eta_array(), ww_arrays(), pp(),
      mus(), Sigmas(), tau() {}

ProductConvolutionOneBatch::ProductConvolutionOneBatch(
    MatrixXd              eta_array,
    std::vector<MatrixXd> ww_arrays,
    MatrixXd              pp,
    MatrixXd              mus,
    std::vector<MatrixXd> Sigmas,
    double tau, double xmin, double xmax, double ymin, double ymax)
        : xmin(xmin), xmax(xmax), ymin(ymin), ymax(ymax),
          eta_array(eta_array), ww_arrays(ww_arrays), pp(pp),
          mus(mus), Sigmas(Sigmas), tau(tau) {}

VectorXd ProductConvolutionOneBatch::compute_entries(const MatrixXd & yy, const MatrixXd & xx) const
{
    int num_batch_points = ww_arrays.size();
    int num_eval_points = xx.rows();

    VectorXd pc_entries(num_eval_points);
    pc_entries.setZero();
    for ( int  i = 0; i < num_batch_points; ++i )
    {
        for ( int k = 0; k < num_eval_points; ++k )
        {
            Vector2d z = pp.row(i) + yy.row(k) - xx.row(k);
            if (point_is_in_ellipsoid(z, mus.row(i), Sigmas[i], tau))
            {
                Vector2d z = pp.row(i) + yy.row(k) - xx.row(k);
                double w_ik = grid_interpolate_at_one_point(xx.row(k), xmin, xmax, ymin, ymax, ww_arrays[i]);
                double phi_ik = grid_interpolate_at_one_point(z, xmin, xmax, ymin, ymax, eta_array);
                pc_entries(k) += w_ik * phi_ik;
            }
        }
    }

    return pc_entries;
}


// ---------------------------------------------------------------
// --------      ProductConvolutionMultipleBatches        --------
// ---------------------------------------------------------------

ProductConvolutionMultipleBatches::ProductConvolutionMultipleBatches(
    std::vector<MatrixXd>              eta_array_batches,
    std::vector<std::vector<MatrixXd>> ww_array_batches,
    std::vector<MatrixXd>              pp_batches,
    std::vector<MatrixXd>              mus_batches,
    std::vector<std::vector<MatrixXd>> Sigmas_batches,
    double tau, double xmin, double xmax, double ymin, double ymax) : pc_batches()
{
    int num_batches = eta_array_batches.size();
    pc_batches.resize(num_batches);
    for (int i = 0; i < num_batches; ++i)
    {
        pc_batches[i] = ProductConvolutionOneBatch(eta_array_batches[i],
                                                   ww_array_batches[i],
                                                   pp_batches[i],
                                                   mus_batches[i],
                                                   Sigmas_batches[i],
                                                   tau, xmin, xmax, ymin, ymax);
    }
}

VectorXd ProductConvolutionMultipleBatches::compute_entries(const MatrixXd & yy, const MatrixXd & xx) const
{
    int num_batches = pc_batches.size();
    int num_eval_points = xx.rows();

    VectorXd pc_entries(num_eval_points);
    pc_entries.setZero();
    for (int i = 0; i < num_batches; ++i)
    {
        pc_entries += pc_batches[i].compute_entries(yy, xx);
    }
    return pc_entries;
}


// ---------------------------------------------------------------
// --------      ProductConvolutionMultipleBatches        --------
// ---------------------------------------------------------------

ProductConvolutionCoeffFn::ProductConvolutionCoeffFn(
    const ProductConvolutionMultipleBatches & pcb, MatrixXd dof_coords) : pcb(pcb), dof_coords(dof_coords) {}

void ProductConvolutionCoeffFn::eval(
    const std::vector< idx_t > &  rowidxs,
    const std::vector< idx_t > &  colidxs,
    real_t *                      matrix ) const
{
    const size_t  n = rowidxs.size();
    const size_t  m = colidxs.size();

    MatrixXd xx(n*m,2);
    MatrixXd yy(n*m,2);
    for (int i = 0; i < n; ++i)
    {
        const idx_t  idxi = rowidxs[ i ];
        for (int j = 0; j < m; ++j)
        {
            const idx_t  idxj = colidxs[ j ];
            xx(j*n+i, 0) = dof_coords(idxj, 0);
            xx(j*n+i, 1) = dof_coords(idxj, 1);

            yy(j*n+i, 0) = dof_coords(idxi, 0);
            yy(j*n+i, 1) = dof_coords(idxi, 1);
        }
    }

    VectorXd eval_values = pcb.compute_entries(yy, xx);

    for ( size_t  j = 0; j < m; ++j )
        {
            for ( size_t  i = 0; i < n; ++i )
            {
                matrix[ j*n + i ] = eval_values(j*n + i);
            }// for
        }// for
}

