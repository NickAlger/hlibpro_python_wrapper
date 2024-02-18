#pragma once

#include <Eigen/Dense>
#include <Eigen/LU>
#include <math.h>

using namespace Eigen;


// Radial basis function interpolation with Gaussian kernel basis functions
double RBF_GAUSS_interpolate( const Eigen::VectorXd & function_at_rbf_points,
                              const Eigen::MatrixXd & rbf_points,
                              const Eigen::VectorXd & eval_point,
                              const double shape_parameter )
{
    int N = rbf_points.cols();

    double function_at_eval_point;
    if ( N == 1 )
    {
        function_at_eval_point = function_at_rbf_points(0);
    }
    else
    {
        Eigen::VectorXd max_pt = rbf_points.rowwise().maxCoeff();
        Eigen::VectorXd min_pt  = rbf_points.rowwise().minCoeff();

        double diam_squared = (max_pt - min_pt).squaredNorm();
        double sigma_squared = diam_squared / shape_parameter*shape_parameter;//(1.0 * 1.0);//(3.0 * 3.0);

        Eigen::MatrixXd M(N, N);
        for ( int jj=0; jj<N; ++jj )
        {
            for ( int ii=0; ii<N; ++ii )
            {
                double r_squared = (rbf_points.col(ii) - rbf_points.col(jj)).squaredNorm();
                M(ii,jj) = exp( - 0.5 * r_squared / sigma_squared);
            }
        }

//        Eigen::VectorXd weights = M.lu().solve(function_at_rbf_points);
        VectorXd weights = M.colPivHouseholderQr().solve(function_at_rbf_points);

        Eigen::VectorXd rbfs_at_eval_point(N);
        for ( int ii=0; ii<N; ++ii )
        {
            double r_squared = (rbf_points.col(ii) - eval_point).squaredNorm();
            rbfs_at_eval_point(ii) = exp( - 0.5 * r_squared / sigma_squared);
        }

        function_at_eval_point = (weights.array() * rbfs_at_eval_point.array()).sum();
    }
    return function_at_eval_point;
}


double tps_interpolate( const VectorXd & function_at_rbf_points,
                        const MatrixXd & rbf_points,
                        const VectorXd & eval_point )
{
    int N = rbf_points.cols();

    double function_at_eval_point;
    if ( N == 1 )
    {
        function_at_eval_point = function_at_rbf_points(0);
    }
    else
    {
        MatrixXd M(N, N);
        for ( int jj=0; jj<N; ++jj )
        {
            for ( int ii=0; ii<N; ++ii )
            {
                if ( ii == jj )
                {
                    M(ii,jj) = 0.0;
                }
                else
                {
                    double r_squared = (rbf_points.col(ii) - rbf_points.col(jj)).squaredNorm();
                    M(ii, jj) = 0.5 * r_squared * log(r_squared);
                }
            }
        }
//        M += 1e-6 * M.norm() * MatrixXd::Identity(N, N);

//        VectorXd weights = M.lu().solve(function_at_rbf_points);
        VectorXd weights = M.colPivHouseholderQr().solve(function_at_rbf_points);

        VectorXd rbfs_at_eval_point(N);
        for ( int ii=0; ii<N; ++ii )
        {
            double r_squared = (rbf_points.col(ii) - eval_point).squaredNorm();
            if ( r_squared == 0.0 )
            {
                rbfs_at_eval_point(ii) = 0.0;
            }
            else
            {
                rbfs_at_eval_point(ii) = 0.5 * r_squared * log(r_squared);
            }
        }

        function_at_eval_point = (weights.array() * rbfs_at_eval_point.array()).sum();
    }
    return function_at_eval_point;
}


VectorXd tps_interpolate_vectorized( const VectorXd & function_at_rbf_points,
                                     const MatrixXd & rbf_points,
                                     const MatrixXd & eval_points )
{
    int num_eval_pts = eval_points.cols();
    VectorXd function_at_eval_points(num_eval_pts);
    for ( int ii=0; ii<num_eval_pts; ++ii )
    {
        function_at_eval_points(ii) = tps_interpolate( function_at_rbf_points,
                                                       rbf_points,
                                                       eval_points.col(ii) );
    }
    return function_at_eval_points;
}

double tps_interpolate_least_squares( const VectorXd & function_at_rbf_points,
                                      const MatrixXd & rbf_points,
                                      const VectorXd & eval_point,
                                      double           regularization_parameter )
{
    int N = rbf_points.cols();

    double function_at_eval_point;
    if ( N == 1 )
    {
        function_at_eval_point = function_at_rbf_points(0);
    }
    else
    {
        MatrixXd PHI(N, N);
        for ( int jj=0; jj<N; ++jj )
        {
            for ( int ii=0; ii<N; ++ii )
            {
                if ( ii == jj )
                {
                    PHI(ii,jj) = 0.0;
                }
                else
                {
                    double r_squared = (rbf_points.col(ii) - rbf_points.col(jj)).squaredNorm();
                    PHI(ii, jj) = 0.5 * r_squared * log(r_squared);
                }
            }
        }

        MatrixXd A(2*N, N);
        A.block(0, 0, N, N) = PHI;
        A.block(N, 0, N, N) = (regularization_parameter * PHI.norm()) * MatrixXd::Identity(N, N);

        VectorXd b(2*N);
        b.head(N) = function_at_rbf_points;
        b.tail(N).setZero();

        VectorXd weights = A.colPivHouseholderQr().solve(b);

//        M += 1e-6 * MatrixXd::Identity(N, N);

//        VectorXd weights = M.lu().solve(function_at_rbf_points);
//        VectorXd weights = M.colPivHouseholderQr().solve(function_at_rbf_points);

        VectorXd rbfs_at_eval_point(N);
        for ( int ii=0; ii<N; ++ii )
        {
            double r_squared = (rbf_points.col(ii) - eval_point).squaredNorm();
            if ( r_squared == 0.0 )
            {
                rbfs_at_eval_point(ii) = 0.0;
            }
            else
            {
                rbfs_at_eval_point(ii) = 0.5 * r_squared * log(r_squared);
            }
        }

        function_at_eval_point = (weights.array() * rbfs_at_eval_point.array()).sum();
    }
    return function_at_eval_point;
}
