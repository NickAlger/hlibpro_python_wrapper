#pragma once

#include <iostream>
#include <math.h>
#include <Eigen/Dense>

#include "kdtree.h"
#include "misc.h"

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

public:
    ProductConvolutionKernelRBF( const vector<Matrix<double, K, 1>> all_points,
                                 const vector<Matrix<double, K, 1>> all_mu,
                                 const vector<Matrix<double, K, K>> all_Sigma,
                                 double                             tau,
                                 const vector<VectorXd>             input_impulse_response_batches,
                                 const vector<int>                  batch_lengths,
                                 double                             rbf_sigma,
                                 const Ref<const Matrix<double, K,   Dynamic>> mesh_vertices,
                                 const Ref<const Matrix<int   , K+1, Dynamic>> mesh_cells ) : mesh(mesh_vertices, mesh_cells)
    {
        num_sample_points = all_points.size();
        sample_points.resize(num_sample_points);
        for ( int ii=0; ii<num_sample_points; ++ii )
        {
            sample_points[ii].point = all_points[ii];
            sample_points[ii].mu = all_mu[ii];
            sample_points[ii].inv_Sigma = all_Sigma[ii].inverse();
        }

        num_batches = input_impulse_response_batches.size();
        impulse_response_batches.resize(num_batches);
        for ( int ii=0; ii<num_batches; ++ii )
        {
            impulse_response_batches[ii] = input_impulse_response_batches[ii];
        }

        tau_squared = tau * tau;

        rbf_sigma_squared = rbf_sigma * rbf_sigma;

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

    VectorXd eval_rbfs_at_point( const Matrix<double, K, 1> & x )
    {
        VectorXd rbfs_at_x(num_sample_points);
        for ( int kk=0; kk<num_sample_points; ++kk )
        {
            double r_squared = (x - sample_points[kk].point).squaredNorm();
            rbfs_at_x(kk) = exp(-0.5 * r_squared / rbf_sigma_squared);
            if (r_squared == 0)
            {
                rbfs_at_x(kk) += 1e-4;
            }

//            if (r_squared > 0)
//            {
//                rbfs_at_x(kk) = 0.5 * r_squared * log(r_squared);
//            }
//            else
//            {
//                rbfs_at_x(kk) = 1e-3;
//            }
        }
        return rbfs_at_x;
    }

    double eval_integral_kernel(const Matrix<double, K, 1> & y, const Matrix<double, K, 1> & x)
    {
        vector<ind_and_coords<K>> all_IC(num_sample_points);
        for ( int kk=0; kk<num_sample_points; ++kk )
        {
            Matrix<double, K, 1> z = y - x + sample_points[kk].point;
            mesh.get_simplex_ind_and_affine_coordinates_of_point( z, all_IC[kk] );
        }

        VectorXd rbfs_at_x = eval_rbfs_at_point( x );
        Matrix<double, Dynamic, 1> weights = interpolation_matrix_factorization.solve(rbfs_at_x);

        vector<int> inside_mesh_inds;
        vector<int> woodbury_inds;
        inside_mesh_inds.reserve(num_sample_points);
        woodbury_inds.reserve(num_sample_points);
        for ( int kk=0; kk<num_sample_points; ++kk )
        {
            if ( all_IC[kk].simplex_ind >= 0 ) // if the point is in the mesh
            {
                inside_mesh_inds.push_back(kk);
            }
            else
            {
                if ( abs(weights(kk)) > 1e-3) // 1e-3
                {
                    woodbury_inds.push_back(kk);
                }
            }
        }

        vector<int> nonzero_phi_inds;
        nonzero_phi_inds.reserve(inside_mesh_inds.size());
        for ( int ii : inside_mesh_inds )
        {
            SamplePoint<K> & SP = sample_points[ii];
            Matrix<double, K, 1> dp = y - x + SP.point - SP.mu;
            if ( dp.transpose() * (SP.inv_Sigma * dp) < tau_squared )
            {
                nonzero_phi_inds.push_back(ii);
            }
        }

        double kernel_entry = 0.0;
        if ( !nonzero_phi_inds.empty() )
        {
            if ( woodbury_inds.size() > 0 )
            {
                pair<MatrixXd, MatrixXd> SDF = submatrix_deletion_factors( interpolation_matrix, woodbury_inds, woodbury_inds );
                woodbury_update(weights, interpolation_matrix, interpolation_matrix_factorization, SDF.first, SDF.second);
            }

            for ( int jj : nonzero_phi_inds )
            {
                double varphi_j_of_y_minus_x = 0.0;
                VectorXd & phi_j = impulse_response_batches[point2batch[jj]];
                ind_and_coords<K> & IC = all_IC[jj];
                for ( int kk=0; kk<K+1; ++kk )
                {
                    varphi_j_of_y_minus_x += IC.affine_coords(kk) * phi_j(mesh.cells(kk, IC.simplex_ind));
                }
                kernel_entry += weights[jj] * varphi_j_of_y_minus_x;
            }
        }
        return kernel_entry;
    }
};
