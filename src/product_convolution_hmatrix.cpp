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


ProductConvolution2d::ProductConvolution2d(std::vector<Vector2d> WW_mins,
                                           std::vector<Vector2d> WW_maxes,
                                           std::vector<MatrixXd> WW_arrays,
                                           std::vector<Vector2d> FF_mins,
                                           std::vector<Vector2d> FF_maxes,
                                           std::vector<MatrixXd> FF_arrays,
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
                                           num_patches(WW_mins.size()),
                                           num_rows(row_coords.rows()),
                                           num_cols(col_coords.rows()),
                                           row_patches(row_coords.rows()),
                                           col_patches(col_coords.rows())
{
    // Compute patches relevant to each row.
    for (int r=0; r < num_rows; ++r)
    {
        const Vector2d y = row_coords.row(r);
        for (int p=0; p < num_patches; ++p)
        {
            const Vector2d y_min = WW_mins[p] + FF_mins[p];
            const Vector2d y_max = WW_maxes[p] + FF_maxes[p];
            if (y_min(0) <= y(0) && y(0) <= y_max(0) &&
                y_min(1) <= y(1) && y(1) <= y_max(1))
            {
                row_patches[r].push_back(p);
            }
        }
        std::sort(row_patches[r].begin(), row_patches[r].end()); // not sure if necessary
    }

    // Compute patches relevant to each col
    for (int c=0; c < num_cols; ++c)
    {
        const Vector2d x = col_coords.row(c);
        for (int p=0; p < num_patches; ++p)
        {
            const Vector2d x_min = WW_mins[p];
            const Vector2d x_max = WW_maxes[p];
            if (x_min(0) <= x(0) && x(0) <= x_max(0) &&
                x_min(1) <= x(1) && x(1) <= x_max(1))
            {
                col_patches[c].push_back(p);
            }
        }
        std::sort(col_patches[c].begin(), col_patches[c].end()); // not sure if necessary
    }
}

VectorXd ProductConvolution2d::get_entries(VectorXi rows, VectorXi cols) const
{
    int num_entries = rows.size();
    std::vector<std::vector<int>> patch_entries(num_patches);
    for (int e=0; e < num_entries; ++e)
    {
        const std::vector<int> pp_row = row_patches[rows(e)];
        const std::vector<int> pp_col = col_patches[cols(e)];
        std::vector<int> pp;
        std::set_intersection(pp_row.begin(), pp_row.end(),
                              pp_col.begin(), pp_col.end(),
                              std::back_inserter(pp));
        for (int i=0; i < pp.size(); ++i)
        {
            patch_entries[pp[i]].push_back(e);
        }
    }

    VectorXd entries(num_entries);
    entries.fill(0.0);
    for (int p=0; p < num_patches; ++p)
    {
        const std::vector<int> ee = patch_entries[p];
        if (ee.size() > 0)
        {
            Array<double, Dynamic, 2> yy;
            Array<double, Dynamic, 2> xx;
            yy.resize(ee.size(), 2);
            xx.resize(ee.size(), 2);
            for (int j=0; j < ee.size(); ++j)
            {
                yy.row(j) = row_coords.row(rows(ee[j]));
                xx.row(j) = col_coords.row(cols(ee[j]));
            }

            VectorXd ww = bilinear_interpolation_regular_grid(xx, WW_mins[p], WW_maxes[p], WW_arrays[p]);
            VectorXd ff = bilinear_interpolation_regular_grid(yy - xx, FF_mins[p], FF_maxes[p], FF_arrays[p]);

            for (int j=0; j < ee.size(); ++j)
            {
                entries(ee[j]) += ww(j) * ff(j);
            }
        }
    }
    return entries;
}

MatrixXd ProductConvolution2d::get_block(VectorXi block_rows, VectorXi block_cols) const
{
    const int num_entries = block_rows.size() * block_cols.size();

    // form row/column index sets that linearly index all entries in the block
    VectorXi rows(num_entries);
    VectorXi cols(num_entries);
    int e = 0;
    for (int j=0; j < block_cols.size(); ++j) // stride down columns first because Eigen uses column-first ordering
    {
        for (int i=0; i < block_rows.size(); ++i)
        {
            rows(e) = block_rows(i);
            cols(e) = block_cols(j);
            e += 1;
        }
    }

    VectorXd entries = ProductConvolution2d::get_entries(rows, cols); // <-- the actual computation of block entries

    // reshape vector of block entries into a matrix
    MatrixXd block_entries(block_rows.size(), block_cols.size());
    e = 0;
    for (int j=0; j < block_cols.size(); ++j)
    {
        for (int i=0; i < block_rows.size(); ++i)
        {
            block_entries(i, j) = entries(e);
            e += 1;
        }
    }
    return block_entries;
}

MatrixXd ProductConvolution2d::get_array() const
{
    VectorXi block_rows(num_rows);
    for (int r=0; r<num_rows; ++r)
    {
        block_rows(r) = r;
    }

    VectorXi block_cols(num_cols);
    for (int c=0; c<num_cols; ++c)
    {
        block_cols(c) = c;
    }

    return ProductConvolution2d::get_block(block_rows, block_cols);
}


// -------------------------------------------------------
// ---------------      PC2DCoeffFn        ---------------
// -------------------------------------------------------

PC2DCoeffFn::PC2DCoeffFn(const ProductConvolution2d & PC) : PC(PC) {}

void PC2DCoeffFn::eval(const std::vector< idx_t > &  rowidxs,
                       const std::vector< idx_t > &  colidxs,
                       real_t *                      matrix ) const
{
    const int nrow = rowidxs.size();
    const int ncol = colidxs.size();
    const int num_entries = nrow * ncol;

    // form row/column index sets that linearly index all entries in the block
    VectorXi rows(num_entries);
    VectorXi cols(num_entries);
    int e = 0;
    for (int j=0; j < ncol; ++j) // stride down columns first because Eigen uses column-first ordering
    {
        for (int i=0; i < nrow; ++i)
        {
            rows(e) = rowidxs[i];
            cols(e) = colidxs[j];
            e += 1;
        }
    }

    VectorXd entries = PC.get_entries(rows, cols); // <-- the actual computation of block entries

    for (int e=0; e<num_entries; ++e)
    {
        matrix[e] = entries(e);
    }
}


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

