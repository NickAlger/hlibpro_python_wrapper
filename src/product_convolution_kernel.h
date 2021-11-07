#pragma once

#include <iostream>
#include <math.h>
#include <Eigen/Dense>

#include "thread-pool-master/thread_pool.hpp"

#include "kdtree.h"
#include "misc.h"
#include "rbf_interpolation.h"

using namespace Eigen;
using namespace std;


template <int K>
struct SamplePoint {Matrix<double, K, 1> point;
                    Matrix<double, K, 1> mu;
                    Matrix<double, K, K> inv_Sigma; };

template <int K>
class ProductConvolutionKernelRBF
{
private:
    SimplexMesh<K>         mesh;
    KDTree<K>              sample_points_kdtree;
    vector<SamplePoint<K>> sample_points;
    vector<VectorXd>       impulse_response_batches;

    vector<int> point2batch;
    int         num_sample_points;
    int         num_batches;
    double      tau_squared;
    double      rbf_sigma_squared;
    int         num_nearest_neighbors;

    thread_pool pool;

public:
    ProductConvolutionKernelRBF( const vector<Matrix<double, K, 1>> all_sample_points,
                                 const vector<Matrix<double, K, 1>> all_sample_mu,
                                 const vector<Matrix<double, K, K>> all_sample_Sigma,
                                 double                             tau,
                                 const vector<VectorXd>             input_impulse_response_batches,
                                 const vector<int>                  batch_lengths,
                                 int                                input_num_nearest_neighbors,
                                 const Ref<const Matrix<double, K,   Dynamic>> mesh_vertices,
                                 const Ref<const Matrix<int   , K+1, Dynamic>> mesh_cells )
                                 : mesh(mesh_vertices, mesh_cells)
    {
        num_nearest_neighbors = input_num_nearest_neighbors;

        num_sample_points = all_sample_points.size();
        sample_points.resize(num_sample_points);
        MatrixXd sample_points_array(K, num_sample_points); // needed for kdtree
        for ( int ii=0; ii<num_sample_points; ++ii )
        {
            sample_points[ii].point = all_sample_points[ii];
            sample_points[ii].mu = all_sample_mu[ii];
            sample_points[ii].inv_Sigma = all_sample_Sigma[ii].inverse(); // Matrix is 2x2 or 3x3, so inverse is OK

            sample_points_array.col(ii) = all_sample_points[ii];
        }

        sample_points_kdtree = KDTree<K>(sample_points_array);

        num_batches = input_impulse_response_batches.size();
        impulse_response_batches.resize(num_batches);
        for ( int ii=0; ii<num_batches; ++ii )
        {
            impulse_response_batches[ii] = input_impulse_response_batches[ii];
        }

        tau_squared = tau * tau;

        point2batch.resize(num_sample_points);
        int ii = 0;
        for ( int bb=0; bb<num_batches; ++bb )
        {
            for ( int dummy=0; dummy<batch_lengths[bb]; ++dummy )
            {
                point2batch[ii] = bb;
                ii += 1;
            }
        }

    }

    vector<pair<Matrix<double,K,1>, double>> interpolation_points_and_values(const Matrix<double, K, 1> & y,
                                                                             const Matrix<double, K, 1> & x)
    {
        pair<VectorXi, VectorXd> nn_result = sample_points_kdtree.nearest_neighbors( x, num_nearest_neighbors );
        VectorXi nearest_inds = nn_result.first;

        int N_nearest = nearest_inds.size();

        vector<ind_and_coords<K>> all_IC(N_nearest);
        vector<bool>              ind_is_good(N_nearest);
        for ( int jj=0; jj<N_nearest; ++jj )
        {
            int ind = nearest_inds[jj];
            Matrix<double, K, 1> z = y - x + sample_points[ind].point;
            mesh.get_simplex_ind_and_affine_coordinates_of_point( z, all_IC[jj] );
            ind_is_good[jj] = ( all_IC[jj].simplex_ind >= 0 ); // y-x+xi is in mesh => varphi_i(y-x) is defined
        }

        vector<pair<Matrix<double,K,1>, double>> good_points_and_values;
        good_points_and_values.reserve(ind_is_good.size());
        for ( int jj=0; jj<N_nearest; ++jj )
        {
            if ( ind_is_good[jj] )
            {
                int ind = nearest_inds[jj];
                SamplePoint<K> & SP = sample_points[ind];
                Matrix<double, K, 1> dp = y - x + SP.point - SP.mu;
                ind_and_coords<K> & IC = all_IC[jj];

                double varphi_at_y_minus_x = 0.0;
                if ( dp.transpose() * (SP.inv_Sigma * dp) < tau_squared )
                {
                    int b = point2batch[ind];
                    VectorXd & phi_j = impulse_response_batches[b];
                    for ( int kk=0; kk<K+1; ++kk )
                    {
                        varphi_at_y_minus_x += IC.affine_coords(kk) * phi_j(mesh.cells(kk, IC.simplex_ind));
                    }
                }
                good_points_and_values.push_back(make_pair(SP.point - x, varphi_at_y_minus_x));
            }
        }
        return good_points_and_values;
    }

    double eval_integral_kernel(const Matrix<double, K, 1> & y, const Matrix<double, K, 1> & x)
    {
        vector<pair<Matrix<double,K,1>, double>> points_and_values   = interpolation_points_and_values(y, x); // forward
        vector<pair<Matrix<double,K,1>, double>> points_and_values_T = interpolation_points_and_values(x, y); // transpose (swap x, y)

        // Add non-duplicates. Inefficient implementation but whatever.
        // Asymptotic complexity not affected because we already have to do O(k^2) matrix operation later anyways
        double tol = 1e-8;
        double tol_squared = tol*tol;

        int N = points_and_values.size();
        int N_T = points_and_values_T.size();
        points_and_values.reserve( N + N_T );
        for ( int ii=0; ii<N_T; ++ii ) // inefficient, but whatever.
        {
            pair<Matrix<double,K,1>, double> & PV_T = points_and_values_T[ii];
            bool is_duplicate = false;
            for ( int jj=0; jj<N; ++jj )
            {
                pair<Matrix<double,K,1>, double> & PV = points_and_values[jj];
                if ( (PV.first - PV_T.first).squaredNorm() < tol_squared )
                {
                    is_duplicate = true;
                    PV.second = 0.5*(PV.second + PV_T.second);
                    break;
                }
            }
            if ( !is_duplicate )
            {
                points_and_values.push_back(PV_T);
            }
        }

        int N_combined = points_and_values.size();
        double kernel_value = 0.0;
        if ( N_combined > 0 )
        {
            MatrixXd P(K, N_combined);
            VectorXd F(N_combined);
            for ( int jj=0; jj<N_combined; ++jj )
            {
                P.col(jj) = points_and_values[jj].first;
                F(jj)     = points_and_values[jj].second;
            }
            kernel_value = tps_interpolate( F, P, MatrixXd::Zero(K,1) );
        }
        return kernel_value;
    }

    MatrixXd eval_integral_kernel_block(const Ref<const Matrix<double, K, Dynamic>> yy,
                                        const Ref<const Matrix<double, K, Dynamic>> xx)
    {
        int nx = xx.cols();
        int ny = yy.cols();
        MatrixXd block(ny, nx);

        auto loop = [&](const int &a, const int &b)
        {
            for ( int ind=a; ind<b; ++ind )
            {
                int jj = ind / ny;
                int ii = ind % ny;
                block(ii, jj) = eval_integral_kernel(yy.col(ii), xx.col(jj));
            }
        };

        pool.parallelize_loop(0, nx * ny, loop);

//        for ( int jj=0; jj<nx; ++jj )
//        {
//            for ( int ii=0; ii<ny; ++ii )
//            {
//                block(ii, jj) = eval_integral_kernel(yy.col(ii), xx.col(jj));
//            }
//        }
        return block;
    }
};


