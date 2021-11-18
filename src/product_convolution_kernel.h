#pragma once

#include <iostream>
#include <math.h>
#include <Eigen/Dense>
#include <hlib.hh>

#include "thread-pool-master/thread_pool.hpp"

#include "kdtree.h"
#include "misc.h"
#include "rbf_interpolation.h"

using namespace Eigen;
using namespace std;

using namespace HLIB;
using namespace SMESH;
//using namespace KDT;

#if HLIB_SINGLE_PREC == 1
using  real_t = float;
#else
using  real_t = double;
#endif

template <int K>
using BatchData = tuple<vector<Matrix<double, K, 1>>, // sample points batch
                        vector<Matrix<double, K, 1>>, // sample mu batch
                        vector<Matrix<double, K, K>>, // sample Sigma batch
                        VectorXd>;                    // impulse response batch

template <int K>
class ImpulseResponseBatches
{
private:

public:
    SimplexMesh                  mesh;
    vector<Matrix<double, K, 1>> pts;
    vector<Matrix<double, K, 1>> mu;
    vector<Matrix<double, K, K>> inv_Sigma;
    vector<VectorXd>             psi_batches;
    vector<int>                  point2batch;
    vector<int>                  batch2point_start;
    vector<int>                  batch2point_stop;

    double                 tau;
    int                    num_neighbors;
    KDTree                 kdtree;

    ImpulseResponseBatches( const Ref<const Matrix<double, K,   Dynamic>> mesh_vertices,
                            const Ref<const Matrix<int   , K+1, Dynamic>> mesh_cells,
                            int                                           num_neighbors,
                            double                                        tau )
        : mesh(mesh_vertices, mesh_cells), num_neighbors(num_neighbors), tau(tau)
    {}

    void build_kdtree()
    {
        MatrixXd pts_matrix(K, pts.size());
        for ( int ii=0; ii<pts.size(); ++ii )
        {
            pts_matrix.col(ii) = pts[ii];
        }
//        kdtree = KDTree(pts_matrix);
        kdtree.build_tree(pts_matrix);
    }

    int num_pts()
    {
        return pts.size();
    }

    int num_batches()
    {
        return psi_batches.size();
    }

    void add_batch( const BatchData<K> & batch_data, bool rebuild_kdtree )
    {
        const vector<Matrix<double, K, 1>> & pts_batch              = get<0>(batch_data);
        const vector<Matrix<double, K, 1>> & mu_batch               = get<1>(batch_data);
        const vector<Matrix<double, K, K>> & Sigma_batch            = get<2>(batch_data);
        const VectorXd                     & impulse_response_batch = get<3>(batch_data);

        int num_new_pts = pts_batch.size();

        batch2point_start.push_back(pts.size());
        int batch_ind = psi_batches.size();
        for ( int ii=0; ii<num_new_pts; ++ii )
        {
            point2batch.push_back( batch_ind );
            pts        .push_back( pts_batch[ii] );
            mu         .push_back( mu_batch[ii] );
            inv_Sigma  .push_back( Sigma_batch[ii].inverse() ); // Matrix is 2x2 or 3x3, so inverse is OK
        }
        batch2point_stop.push_back(pts.size());

        psi_batches.push_back(impulse_response_batch);

        if ( rebuild_kdtree )
        {
            build_kdtree();
        }
    }

    vector<pair<Matrix<double,K,1>, double>> interpolation_points_and_values(const Matrix<double, K, 1> & y,
                                                                             const Matrix<double, K, 1> & x) const
    {
        pair<VectorXi, VectorXd> nn_result = kdtree.query( x, num_neighbors );
        VectorXi nearest_inds = nn_result.first;

        int N_nearest = nearest_inds.size();

        vector<int>      all_simplex_inds (N_nearest);
        vector<VectorXd> all_affine_coords(N_nearest);
        vector<bool>     ind_is_good      (N_nearest);
        for ( int jj=0; jj<N_nearest; ++jj )
        {
            int ind = nearest_inds(jj);
            Matrix<double, K, 1> z = y - x + pts[ind];
            std::pair<int,VectorXd> IC = mesh.point_query( z );
            all_simplex_inds[jj]  = IC.first;
            all_affine_coords[jj] = IC.second;
            ind_is_good[jj] = ( all_simplex_inds[jj] >= 0 ); // y-x+xi is in mesh => varphi_i(y-x) is defined
        }

        vector<pair<Matrix<double,K,1>, double>> good_points_and_values;
        good_points_and_values.reserve(ind_is_good.size());
        for ( int jj=0; jj<N_nearest; ++jj )
        {
            if ( ind_is_good[jj] )
            {
                int ind = nearest_inds[jj];
                Matrix<double, K, 1> dp = y - x + pts[ind] - mu[ind];

                double varphi_at_y_minus_x = 0.0;
                if ( dp.transpose() * (inv_Sigma[ind] * dp) < tau*tau )
                {
                    int b = point2batch[ind];
                    const VectorXd & phi_j = psi_batches[b];
                    for ( int kk=0; kk<K+1; ++kk )
                    {
                        varphi_at_y_minus_x += all_affine_coords[jj](kk) * phi_j(mesh.cells(kk, all_simplex_inds[jj]));
                    }
                }
                good_points_and_values.push_back(make_pair(pts[ind] - x, varphi_at_y_minus_x));
            }
        }
        return good_points_and_values;
    }

};

template <int K>
class ProductConvolutionKernelRBF : public TCoeffFn< real_t >
{
private:
    shared_ptr<ImpulseResponseBatches<K>> col_batches;
    shared_ptr<ImpulseResponseBatches<K>> row_batches;

public:
    vector<Matrix<double, K, 1>> row_coords;
    vector<Matrix<double, K, 1>> col_coords;
    double                       gamma;

    thread_pool pool;

    ProductConvolutionKernelRBF( shared_ptr<ImpulseResponseBatches<K>> col_batches,
                                 shared_ptr<ImpulseResponseBatches<K>> row_batches,
                                 vector<Matrix<double, K, 1>>          col_coords,
                                 vector<Matrix<double, K, 1>>          row_coords,
                                 double                                gamma )
        : col_batches(col_batches),
          row_batches(row_batches),
          row_coords(row_coords),
          col_coords(col_coords),
          gamma(gamma)
    {}

    double eval_integral_kernel(const Matrix<double, K, 1> & y, const Matrix<double, K, 1> & x ) const
    {
        vector<pair<Matrix<double,K,1>, double>> points_and_values_FWD;
        if ( col_batches->num_pts() > 0 )
        {
            points_and_values_FWD = col_batches->interpolation_points_and_values(y, x); // forward
        }

        vector<pair<Matrix<double,K,1>, double>> points_and_values_ADJ;
        if ( row_batches->num_pts() > 0 )
        {
            points_and_values_ADJ = row_batches->interpolation_points_and_values(x, y); // adjoint (swap x, y)
        }

//        points_and_values_FWD.insert(points_and_values_FWD.begin(),
//                                     points_and_values_ADJ.begin(),
//                                     points_and_values_ADJ.end());

        // Add non-duplicates. Inefficient implementation but whatever.
        // Asymptotic complexity not affected because we already have to do O(k^2) matrix operation later anyways
        double tol = 1e-7;
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
//            kernel_value = tps_interpolate( F, P, MatrixXd::Zero(K,1) );
            kernel_value = tps_interpolate_least_squares( F, P, MatrixXd::Zero(K,1), gamma );
        }
        return kernel_value;
    }

    MatrixXd eval_integral_kernel_block(const Ref<const Matrix<double, K, Dynamic>> yy,
                                        const Ref<const Matrix<double, K, Dynamic>> xx )
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

    void eval  ( const std::vector< idx_t > &  rowidxs,
                         const std::vector< idx_t > &  colidxs,
                         real_t *                      matrix ) const
    {
        int nrow = rowidxs.size();
        int ncol = colidxs.size();

        for ( size_t  jj = 0; jj < ncol; ++jj )
        {
            for ( size_t  ii = 0; ii < nrow; ++ii )
            {
                matrix[ jj*nrow + ii ] = eval_integral_kernel(row_coords[rowidxs[ii]], col_coords[colidxs[jj]]);
            }
        }
    }

    using TCoeffFn< real_t >::eval;

    virtual matform_t  matrix_format  () const { return MATFORM_NONSYM; }

};


