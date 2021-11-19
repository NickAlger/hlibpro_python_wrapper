#pragma once

#include <iostream>
#include <math.h>
#include <Eigen/Dense>
#include <hlib.hh>

#include "thread_pool.hpp"

#include "kdtree.h"
#include "misc.h"
#include "rbf_interpolation.h"

using namespace Eigen;
using namespace std;

using namespace HLIB;
using namespace SMESH;
using namespace KDT;

#if HLIB_SINGLE_PREC == 1
using  real_t = float;
#else
using  real_t = double;
#endif


class ImpulseResponseBatches
{
private:

public:
    int                     dim;
    SimplexMesh             mesh;
    vector<Eigen::VectorXd> pts;
    vector<Eigen::VectorXd> mu;
    vector<Eigen::MatrixXd> inv_Sigma;
    vector<VectorXd>        psi_batches;
    vector<int>             point2batch;
    vector<int>             batch2point_start;
    vector<int>             batch2point_stop;

    double                 tau;
    int                    num_neighbors;
    KDTree                 kdtree;

    ImpulseResponseBatches( const Eigen::Ref<const Eigen::MatrixXd> mesh_vertices, // shape=(dim, num_vertices)
                            const Eigen::Ref<const Eigen::MatrixXi> mesh_cells,    // shape=(dim+1, num_cells)
                            int                                     num_neighbors,
                            double                                  tau )
        : mesh(mesh_vertices, mesh_cells), num_neighbors(num_neighbors), tau(tau)
    {
        dim = mesh_vertices.rows();
    }

    void build_kdtree()
    {
        Eigen::MatrixXd pts_matrix(dim, num_pts());
        for ( int ii=0; ii<pts.size(); ++ii )
        {
            pts_matrix.col(ii) = pts[ii];
        }
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

    void add_batch( const std::vector<Eigen::VectorXd> batch_points,
                    const std::vector<Eigen::VectorXd> batch_mu,
                    const std::vector<Eigen::MatrixXd> batch_Sigma,
                    const Eigen::VectorXd &            impulse_response_batch,
                    bool                               rebuild_kdtree )
    {
        int num_new_pts = batch_points.size();

        batch2point_start.push_back(num_pts());
        int batch_ind = psi_batches.size();
        for ( int ii=0; ii<num_new_pts; ++ii )
        {
            point2batch.push_back( batch_ind );
            pts        .push_back( batch_points[ii] );
            mu         .push_back( batch_mu[ii] );
            inv_Sigma  .push_back( batch_Sigma[ii].inverse() ); // Matrix is 2x2 or 3x3, so inverse is OK
        }
        batch2point_stop.push_back(num_pts());

        psi_batches.push_back(impulse_response_batch);

        if ( rebuild_kdtree )
        {
            build_kdtree();
        }
    }

    std::vector<std::pair<Eigen::VectorXd, double>> interpolation_points_and_values(const Eigen::VectorXd & y,
                                                                                    const Eigen::VectorXd & x) const
    {
        pair<Eigen::VectorXi, Eigen::VectorXd> nn_result = kdtree.query( x, num_neighbors );
        VectorXi nearest_inds = nn_result.first;

        int N_nearest = nearest_inds.size();

        vector<int>      all_simplex_inds (N_nearest);
        vector<VectorXd> all_affine_coords(N_nearest);
        vector<bool>     ind_is_good      (N_nearest);
        for ( int jj=0; jj<N_nearest; ++jj )
        {
            int ind = nearest_inds(jj);
            Eigen::VectorXd z = y - x + pts[ind];
            std::pair<Eigen::VectorXi, Eigen::MatrixXd> IC = mesh.first_point_collision( z );
            all_simplex_inds[jj]  = IC.first(0);
            all_affine_coords[jj] = IC.second.col(0);
            ind_is_good[jj] = ( all_simplex_inds[jj] >= 0 ); // y-x+xi is in mesh => varphi_i(y-x) is defined
        }

        std::vector<std::pair<Eigen::VectorXd, double>> good_points_and_values;
        good_points_and_values.reserve(ind_is_good.size());
        for ( int jj=0; jj<N_nearest; ++jj )
        {
            if ( ind_is_good[jj] )
            {
                int ind = nearest_inds[jj];
                VectorXd dp = y - x + pts[ind] - mu[ind];

                double varphi_at_y_minus_x = 0.0;
                if ( dp.transpose() * (inv_Sigma[ind] * dp) < tau*tau )
                {
                    int b = point2batch[ind];
                    const VectorXd & phi_j = psi_batches[b];
                    for ( int kk=0; kk<dim+1; ++kk )
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


class ProductConvolutionKernelRBF : public TCoeffFn< real_t >
{
private:
    int dim;
    shared_ptr<ImpulseResponseBatches> col_batches;
    shared_ptr<ImpulseResponseBatches> row_batches;

public:
    std::vector<Eigen::VectorXd> row_coords;
    std::vector<Eigen::VectorXd> col_coords;
    double                       gamma;

    thread_pool pool;

    ProductConvolutionKernelRBF( shared_ptr<ImpulseResponseBatches> col_batches,
                                 shared_ptr<ImpulseResponseBatches> row_batches,
                                 vector<Eigen::VectorXd>            col_coords,
                                 vector<Eigen::VectorXd>            row_coords,
                                 double                             gamma )
        : col_batches(col_batches),
          row_batches(row_batches),
          row_coords(row_coords),
          col_coords(col_coords),
          gamma(gamma)
    {
        dim = col_batches->dim;
    }

    double eval_integral_kernel(const Eigen::VectorXd & y,
                                const Eigen::VectorXd & x ) const
    {
        std::vector<std::pair<Eigen::VectorXd, double>> points_and_values_FWD;
        if ( col_batches->num_pts() > 0 )
        {
            points_and_values_FWD = col_batches->interpolation_points_and_values(y, x); // forward
        }

        std::vector<std::pair<Eigen::VectorXd, double>> points_and_values_ADJ;
        if ( row_batches->num_pts() > 0 )
        {
            points_and_values_ADJ = row_batches->interpolation_points_and_values(x, y); // adjoint (swap x, y)
        }

        // Add non-duplicates. Inefficient implementation but whatever.
        // Asymptotic complexity not affected because we already have to do O(k^2) matrix operation later anyways
        double tol = 1e-7;
        double tol_squared = tol*tol;

        int N_FWD = points_and_values_FWD.size();
        int N_ADJ = points_and_values_ADJ.size();

        points_and_values_FWD.reserve( N_FWD + N_ADJ );
        for ( int ii=0; ii<N_ADJ; ++ii )
        {
            std::pair<Eigen::VectorXd, double> & PV_ADJ = points_and_values_ADJ[ii];
            bool is_duplicate = false;
            for ( int jj=0; jj<N_FWD; ++jj )
            {
                pair<Eigen::VectorXd, double> & PV_FWD = points_and_values_FWD[jj];
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
            Eigen::MatrixXd P(dim, N_combined);
            Eigen::VectorXd F(N_combined);
            for ( int jj=0; jj<N_combined; ++jj )
            {
                P.col(jj) = points_and_values_FWD[jj].first;
                F(jj)     = points_and_values_FWD[jj].second;
            }
            kernel_value = tps_interpolate_least_squares( F, P, Eigen::MatrixXd::Zero(dim,1), gamma );
        }
        return kernel_value;
    }

    inline double eval_matrix_entry(const int row_ind,
                                    const int col_ind ) const
    {
        return eval_integral_kernel(row_coords[row_ind], col_coords[col_ind]);
    }

    Eigen::MatrixXd eval_integral_kernel_block(const Eigen::Ref<const Eigen::MatrixXd> yy,  // shape=(dim,num_y)
                                               const Eigen::Ref<const Eigen::MatrixXd> xx ) // shape=(dim,num_x)
    {
        int nx = xx.cols();
        int ny = yy.cols();
        Eigen::MatrixXd block(ny, nx);

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
//                matrix[ jj*nrow + ii ] = eval_integral_kernel(row_coords[rowidxs[ii]], col_coords[colidxs[jj]]);
                matrix[ jj*nrow + ii ] = eval_matrix_entry(rowidxs[ii], colidxs[jj]);
            }
        }
    }

    using TCoeffFn< real_t >::eval;

    virtual matform_t  matrix_format  () const { return MATFORM_NONSYM; }

};


