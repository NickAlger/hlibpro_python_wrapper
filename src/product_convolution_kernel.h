#pragma once

#include <iostream>
#include <math.h>
#include <Eigen/Dense>

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

    double eval_integral_kernel(const Matrix<double, K, 1> & y, const Matrix<double, K, 1> & x)
    {
        pair<VectorXi, VectorXd> nn_result = sample_points_kdtree.nearest_neighbors( x, num_nearest_neighbors );
        VectorXi nearest_sample_point_inds = nn_result.first;

        int N_nearest = nearest_sample_point_inds.size();

        vector<int> good_sample_point_inds;
        good_sample_point_inds.reserve(N_nearest);

        vector<ind_and_coords<K>> good_IC;
        good_IC.reserve(N_nearest);

        for ( int jj=0; jj<N_nearest; ++jj )
        {
            int sample_ind = nearest_sample_point_inds[jj];
            Matrix<double, K, 1> z = y - x + sample_points[sample_ind].point;
            ind_and_coords<K> IC;
            mesh.get_simplex_ind_and_affine_coordinates_of_point( z, IC );
            if ( IC.simplex_ind >= 0 ) // y-x+xi is in mesh => varphi(y-x) is defined
            {
                good_sample_point_inds.push_back(sample_ind);
                good_IC.push_back(IC);
            }
        }

        int N_good = good_sample_point_inds.size();

        double varphi_at_y_minus_x = 0.0;
        if ( N_good > 0 )
        {
            MatrixXd good_sample_points(K, N_good);
            VectorXd good_varphis_at_y_minus_x(N_good);
            good_varphis_at_y_minus_x.setZero();

            for ( int jj=0; jj<N_good; ++jj )
            {
                int sample_ind = good_sample_point_inds[jj];
                SamplePoint<K> & SP = sample_points[sample_ind];
                good_sample_points.col(jj) = SP.point;

                Matrix<double, K, 1> dp = y - x + SP.point - SP.mu;
                if ( dp.transpose() * (SP.inv_Sigma * dp) < tau_squared )
                {
                    int b = point2batch[sample_ind];
                    VectorXd & phi_j = impulse_response_batches[b];
                    ind_and_coords<K> & IC = good_IC[jj];
                    for ( int kk=0; kk<K+1; ++kk )
                    {
                        good_varphis_at_y_minus_x(jj) += IC.affine_coords(kk) * phi_j(mesh.cells(kk, IC.simplex_ind));
                    }
                }
            }
            varphi_at_y_minus_x = tps_interpolate( good_varphis_at_y_minus_x, good_sample_points, x );
        }

        return varphi_at_y_minus_x;
    }

    MatrixXd eval_integral_kernel_block(const Ref<const Matrix<double, K, Dynamic>> yy,
                                        const Ref<const Matrix<double, K, Dynamic>> xx)
    {
        int nx = xx.cols();
        int ny = yy.cols();
        MatrixXd block(ny, nx);
        for ( int jj=0; jj<nx; ++jj )
        {
            for ( int ii=0; ii<ny; ++ii )
            {
                block(ii, jj) = eval_integral_kernel(yy.col(ii), xx.col(jj));
            }
        }
        return block;
    }
};
