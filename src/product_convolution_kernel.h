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
using BatchData = tuple<vector<Matrix<double, K, 1>>, // sample points batch
                        vector<Matrix<double, K, 1>>, // sample mu batch
                        vector<Matrix<double, K, K>>, // sample Sigma batch
                        VectorXd>;                    // impulse response batch

template <int K>
class ImpulseResponseBatches
{
private:
    SimplexMesh<K>               mesh;
    vector<Matrix<double, K, 1>> pts;
    vector<Matrix<double, K, 1>> mu;
    vector<Matrix<double, K, K>> inv_Sigma;
    vector<VectorXd>             psi_batches;
    vector<int>                  point2batch;

    bool rebuild_kdtree = true;

public:
    double                 tau;
    int                    num_neighbors;
    KDTree<K>              kdtree;

    ImpulseResponseBatches( const Ref<const Matrix<double, K,   Dynamic>> & mesh_vertices,
                            const Ref<const Matrix<int   , K+1, Dynamic>> & mesh_cells,
                            int                                             num_neighbors,
                            double                                          tau )
        : mesh(mesh_vertices, mesh_cells), num_neighbors(num_neighbors), tau(tau)
    {}

    void build_kdtree()
    {
        kdtree = KDTree<K>(pts);
    }

    void add_batch( const BatchData<K> & batch_data, bool rebuild_kdtree )
    {
        const vector<Matrix<double, K, 1>> & pts_batch              = get<0>(batch_data);
        const vector<Matrix<double, K, 1>> & mu_batch               = get<1>(batch_data);
        const vector<Matrix<double, K, K>> & Sigma_batch            = get<2>(batch_data);
        const VectorXd                     & impulse_response_batch = get<3>(batch_data);

        int num_new_pts = pts_batch.size();

        int batch_ind = psi_batches.size();
        for ( int ii=0; ii<num_new_pts; ++ii )
        {
            point2batch.push_back( batch_ind );
            pts        .push_back( pts_batch[ii] );
            mu         .push_back( mu_batch[ii] );
            inv_Sigma  .push_back( Sigma_batch[ii].inverse() ); // Matrix is 2x2 or 3x3, so inverse is OK
        }

        psi_batches.push_back(impulse_response_batch);

        if ( rebuild_kdtree )
        {
            build_kdtree();
        }
    }

    vector<pair<Matrix<double,K,1>, double>> interpolation_points_and_values(const Matrix<double, K, 1> & y,
                                                                             const Matrix<double, K, 1> & x)
    {
        pair<VectorXi, VectorXd> nn_result = kdtree.nearest_neighbors( x, num_neighbors );
        VectorXi nearest_inds = nn_result.first;

        int N_nearest = nearest_inds.size();

        vector<ind_and_coords<K>> all_IC(N_nearest);
        vector<bool>              ind_is_good(N_nearest);
        for ( int jj=0; jj<N_nearest; ++jj )
        {
            int ind = nearest_inds[jj];
            Matrix<double, K, 1> z = y - x + pts[ind];
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
                Matrix<double, K, 1> dp = y - x + pts[ind] - mu[ind];
                ind_and_coords<K> & IC = all_IC[jj];

                double varphi_at_y_minus_x = 0.0;
                if ( dp.transpose() * (inv_Sigma[ind] * dp) < tau*tau )
                {
                    int b = point2batch[ind];
                    VectorXd & phi_j = psi_batches[b];
                    for ( int kk=0; kk<K+1; ++kk )
                    {
                        varphi_at_y_minus_x += IC.affine_coords(kk) * phi_j(mesh.cells(kk, IC.simplex_ind));
                    }
                }
                good_points_and_values.push_back(make_pair(pts[ind] - x, varphi_at_y_minus_x));
            }
        }
        return good_points_and_values;
    }

};

template <int K>
class ProductConvolutionKernelRBF
{
private:
    ImpulseResponseBatches<K> IRO_FWD;
    ImpulseResponseBatches<K> IRO_ADJ;

public:
    thread_pool pool;

    ProductConvolutionKernelRBF( const vector<BatchData<K>>                    & all_batches_data_FWD,
                                 const Ref<const Matrix<double, K,   Dynamic>> & mesh_vertices_FWD,
                                 const Ref<const Matrix<int   , K+1, Dynamic>> & mesh_cells_FWD,
                                 int                                             num_neighbors_FWD,
                                 double                                          tau_FWD,

                                 const vector<BatchData<K>>                    & all_batches_data_ADJ,
                                 const Ref<const Matrix<double, K,   Dynamic>> & mesh_vertices_ADJ,
                                 const Ref<const Matrix<int   , K+1, Dynamic>> & mesh_cells_ADJ,
                                 int                                             num_neighbors_ADJ,
                                 double                                          tau_ADJ)
        : IRO_FWD(mesh_vertices_FWD,
                  mesh_cells_FWD,
                  num_neighbors_FWD,
                  tau_FWD),
          IRO_ADJ(mesh_vertices_ADJ,
                  mesh_cells_ADJ,
                  num_neighbors_ADJ,
                  tau_ADJ)
    {
        for ( BatchData<K> batch_data : all_batches_data_FWD )
        {
            IRO_FWD.add_batch(batch_data, false);
        }
        IRO_FWD.build_kdtree();

        for ( BatchData<K> batch_data : all_batches_data_ADJ )
        {
            IRO_ADJ.add_batch(batch_data, false);
        }
        IRO_ADJ.build_kdtree();
    }

    void set_tau_FWD(double new_tau)
    {
        IRO_FWD.tau = new_tau;
    }

    void set_tau_ADJ(double new_tau)
    {
        IRO_ADJ.tau = new_tau;
    }

    void set_tau(double new_tau)
    {
        set_tau_FWD(new_tau);
        set_tau_ADJ(new_tau);
    }

    void set_num_neighbors_FWD(int new_num_neighbors)
    {
        IRO_FWD.num_neighbors = new_num_neighbors;
    }

    void set_num_neighbors_ADJ(int new_num_neighbors)
    {
        IRO_ADJ.num_neighbors = new_num_neighbors;
    }

    void set_num_neighbors(int new_num_neighbors)
    {
        set_num_neighbors_FWD(new_num_neighbors);
        set_num_neighbors_ADJ(new_num_neighbors);
    }

    void add_batch_FWD( const BatchData<K> & batch_data, bool rebuild_kdtree )
    {
        IRO_FWD.add_batch(batch_data, rebuild_kdtree);
    }

    void add_batch_ADJ( const BatchData<K> & batch_data, bool rebuild_kdtree )
    {
        IRO_ADJ.add_batch(batch_data, rebuild_kdtree);
    }

    double eval_integral_kernel(const Matrix<double, K, 1> & y, const Matrix<double, K, 1> & x)
    {
        vector<pair<Matrix<double,K,1>, double>> points_and_values_FWD
            = IRO_FWD.interpolation_points_and_values(y, x); // forward

        vector<pair<Matrix<double,K,1>, double>> points_and_values_ADJ
            = IRO_ADJ.interpolation_points_and_values(x, y); // adjoint (swap x, y)

        // Add non-duplicates. Inefficient implementation but whatever.
        // Asymptotic complexity not affected because we already have to do O(k^2) matrix operation later anyways
        double tol = 1e-8;
        double tol_squared = tol*tol;

        int N_FWD = points_and_values_FWD.size();
        int N_ADJ = points_and_values_ADJ.size();

        points_and_values_FWD.reserve( N_FWD + N_ADJ );
        for ( int ii=0; ii<N_ADJ; ++ii )
        {
            pair<Matrix<double,K,1>, double> & PV_ADJ = points_and_values_ADJ[ii];
            bool is_duplicate = false;
            for ( int jj=0; jj<N_FWD; ++jj )
            {
                pair<Matrix<double,K,1>, double> & PV_FWD = points_and_values_FWD[jj];
                if ( (PV_FWD.first - PV_ADJ.first).squaredNorm() < tol_squared )
                {
                    is_duplicate = true;
                    PV_FWD.first =  0.5*(PV_FWD.first  + PV_ADJ.first);
                    PV_FWD.second = 0.5*(PV_FWD.second + PV_ADJ.second);
                    break;
                }
            }
            if ( !is_duplicate )
            {
                points_and_values_FWD.push_back(PV_ADJ);
            }
        }

        int N_combined = points_and_values_FWD.size();
        double kernel_value = 0.0;
        if ( N_combined > 0 )
        {
            MatrixXd P(K, N_combined);
            VectorXd F(N_combined);
            for ( int jj=0; jj<N_combined; ++jj )
            {
                P.col(jj) = points_and_values_FWD[jj].first;
                F(jj)     = points_and_values_FWD[jj].second;
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

        return block;
    }
};


