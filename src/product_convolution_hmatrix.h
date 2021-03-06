#pragma once

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

//#include "grid_interpolate.h"

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


bool point_is_in_ellipsoid(VectorXd z, VectorXd mu, MatrixXd Sigma, double tau);


class ProductConvolution2d
{
private:
    std::vector<Vector2d> WW_mins;
    std::vector<Vector2d> WW_maxes;
    std::vector<MatrixXd> WW_arrays;

    std::vector<Vector2d> FF_mins;
    std::vector<Vector2d> FF_maxes;
    std::vector<MatrixXd> FF_arrays;

    Matrix<double, Dynamic, 2> row_coords;
    Matrix<double, Dynamic, 2> col_coords;

    int num_patches;
    int num_rows;
    int num_cols;

    std::vector<std::vector<int>> row_patches;
    std::vector<std::vector<int>> col_patches;

public:
    ProductConvolution2d(std::vector<Vector2d> WW_mins,
                         std::vector<Vector2d> WW_maxes,
                         std::vector<MatrixXd> WW_arrays,
                         std::vector<Vector2d> FF_mins,
                         std::vector<Vector2d> FF_maxes,
                         std::vector<MatrixXd> FF_arrays,
                         Matrix<double, Dynamic, 2> row_coords,
                         Matrix<double, Dynamic, 2> col_coords);

    VectorXd get_entries(VectorXi rows, VectorXi cols) const;
    MatrixXd get_block(VectorXi block_rows, VectorXi block_cols) const;
    MatrixXd get_array() const;
};


class PC2DCoeffFn : public TCoeffFn< real_t >
{
private:
    ProductConvolution2d PC;

public:
    PC2DCoeffFn (const ProductConvolution2d & PC);

    virtual void eval  ( const std::vector< idx_t > &  rowidxs,
                         const std::vector< idx_t > &  colidxs,
                         real_t *                      matrix ) const;

    using TCoeffFn< real_t >::eval;

//    virtual matform_t  matrix_format  () const { return MATFORM_SYM; }
    virtual matform_t  matrix_format  () const { return MATFORM_NONSYM; }
};


class ProductConvolutionOneBatch
{
private:
    double xmin;
    double xmax;
    double ymin;
    double ymax;
    MatrixXd eta_array;
    std::vector<MatrixXd> ww_arrays;
    MatrixXd pp;
    MatrixXd mus;
    std::vector<MatrixXd> Sigmas;
    double tau;

public:
    ProductConvolutionOneBatch();

    ProductConvolutionOneBatch(MatrixXd eta_array,
                               std::vector<MatrixXd> ww_arrays,
                               MatrixXd pp,
                               MatrixXd mus,
                               std::vector<MatrixXd> Sigmas,
                               double tau, double xmin, double xmax, double ymin, double ymax);

    VectorXd compute_entries(const MatrixXd & yy, const MatrixXd & xx) const;
};


class ProductConvolutionMultipleBatches
{
private:
    std::vector<ProductConvolutionOneBatch> pc_batches;

public:
    ProductConvolutionMultipleBatches(std::vector<MatrixXd> eta_array_batches,
                                      std::vector<std::vector<MatrixXd>> ww_array_batches,
                                      std::vector<MatrixXd> pp_batches,
                                      std::vector<MatrixXd> mus_batches,
                                      std::vector<std::vector<MatrixXd>> Sigmas_batches,
                                      double tau, double xmin, double xmax, double ymin, double ymax);

    VectorXd compute_entries(const MatrixXd & yy, const MatrixXd & xx) const;
};


class ProductConvolutionCoeffFn : public TCoeffFn< real_t >
{
private:
    ProductConvolutionMultipleBatches pcb;
    MatrixXd dof_coords;

public:
    ProductConvolutionCoeffFn (const ProductConvolutionMultipleBatches & pcb, MatrixXd dof_coords);

    virtual void eval  ( const std::vector< idx_t > &  rowidxs,
                         const std::vector< idx_t > &  colidxs,
                         real_t *                      matrix ) const;

    using TCoeffFn< real_t >::eval;

//    virtual matform_t  matrix_format  () const { return MATFORM_SYM; }
    virtual matform_t  matrix_format  () const { return MATFORM_NONSYM; }
};


//void init_product_convolution_hmatrix_bindings(py::module &m);