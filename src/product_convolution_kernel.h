#pragma once

#include <iostream>
#include <math.h>
#include <Eigen/Dense>
#include <hlib.hh>

#include "thread_pool.hpp"

#include "kdtree.h"
#include "rbf_interpolation.h"


namespace PCK {

#if HLIB_SINGLE_PREC == 1
using  real_t = float;
#else
using  real_t = double;
#endif

class ImpulseResponseBatches
{
private:

public:
    int                          dim;
    SMESH::SimplexMesh           mesh;
    std::vector<Eigen::VectorXd> pts;
    std::vector<double>          vol;
    std::vector<Eigen::VectorXd> mu;
    std::vector<Eigen::MatrixXd> inv_Sigma;
    std::vector<Eigen::VectorXd> psi_batches;
    std::vector<int>             point2batch;
    std::vector<int>             batch2point_start;
    std::vector<int>             batch2point_stop;

    double                  tau;
    int                     num_neighbors;
    KDT::KDTree             kdtree;

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

    int num_pts() const
    {
        return pts.size();
    }

    int num_batches() const
    {
        return psi_batches.size();
    }

    void add_batch( const std::vector<Eigen::VectorXd> batch_points,
                    const std::vector<double>          batch_vol,
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
            vol        .push_back( batch_vol[ii] );
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
        std::pair<Eigen::VectorXi, Eigen::MatrixXd> IC_x = mesh.first_point_collision( x );
        int simplex_ind_x               = IC_x.first(0);
        Eigen::VectorXd affine_coords_x = IC_x.second.col(0);

        double   vol_at_x = 0.0;
        Eigen::VectorXd mu_at_x(dim);
        mu_at_x.setZero();
        if ( simplex_ind_x >= 0 ) // if x is in the mesh
        {
            for ( int kk=0; kk<dim+1; ++kk )
            {
                int vertex_ind = mesh.cells(kk, simplex_ind_x);
                vol_at_x += affine_coords_x(kk) * vol[vertex_ind]; // BAD
//                mu_at_x  += affine_coords_x(kk) *  mu[vertex_ind];
//                mu_at_x  += mu[vertex_ind];
//                std::cout << std::endl;
//                std::cout << "mu_at_x:" << mu_at_x << std::endl;
//                std::cout << "vertex_ind:" << vertex_ind << std::endl;
//                std::cout << "mu.size():" << mu.size() << std::endl;
            }
        }

        pair<Eigen::VectorXi, Eigen::VectorXd> nn_result = kdtree.query( x, min(num_neighbors, num_pts()) );
        Eigen::VectorXi nearest_inds = nn_result.first;

        int N_nearest = nearest_inds.size();

        std::vector<int>             all_simplex_inds (N_nearest);
        std::vector<Eigen::VectorXd> all_affine_coords(N_nearest);
        std::vector<bool>            ind_is_good      (N_nearest);
        for ( int jj=0; jj<N_nearest; ++jj )
        {
            int ind = nearest_inds(jj);
            Eigen::VectorXd z = y - x + pts[ind]; // C
//            Eigen::VectorXd mu_at_xj = mu[ind]; // D
//            Eigen::VectorXd z = y - mu_at_x + mu_at_xj; // D
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
                double vol_at_xj = vol[ind];
                Eigen::VectorXd dp = y - x + pts[ind] - mu[ind]; // C
//                Eigen::VectorXd dp = y - mu_at_x; // D
                double varphi_at_y_minus_x = 0.0;
                if ( dp.transpose() * (inv_Sigma[ind] * dp) < tau*tau )
                {
                    int b = point2batch[ind];
                    const Eigen::VectorXd & phi_j = psi_batches[b];
                    for ( int kk=0; kk<dim+1; ++kk )
                    {
                        varphi_at_y_minus_x += vol_at_xj * all_affine_coords[jj](kk) * phi_j(mesh.cells(kk, all_simplex_inds[jj])); // A
//                        varphi_at_y_minus_x += all_affine_coords[jj](kk) * phi_j(mesh.cells(kk, all_simplex_inds[jj])); // B
                    }
                }
//                varphi_at_y_minus_x *= vol_at_x; // B
                good_points_and_values.push_back(make_pair(pts[ind] - x, varphi_at_y_minus_x));
            }
        }
        return good_points_and_values;
    }

};

//struct EllipsoidData
//{
//    const Eigen::VectorXd vol_at_mesh_nodes; // shape=(N,)
//    const Eigen::MatrixXd mu_at_mesh_nodes; // shape=(d,N)
//    const Eigen::MatrixXd Sigma_at_mesh_nodes; // shape=(d*d,N)
//    const Eigen::VectorXd tau_at_mesh_nodes; // shape=(N,)
//}

struct Ellipsoid
{
    // Ellipsoid is the set of all points p such that
        //   (p - mu)' * inv(Sigma) * (p - mu) < tau^2
    const double          vol;
    const Eigen::VectorXd mu; // shape=(d,)
    const Eigen::MatrixXd Sigma; // shape=(d,d)
    const double          tau
};

bool point_is_in_ellipsoid(const Eigen::VectorXd & p, const Ellipsoid & E)
{
    Eigen::VectorXd dp = p - E.mu;
    return dp.transpose() * E.Sigma.ldlt().solve(dp) < E.tau*E.tau;
}

Ellipsoid get_ellipsoid_at_point(const Eigen::VectorXd    & p,
                                 const vector<Ellipsoid>  & ellipsoids_at_mesh_nodes,
                                 const SMESH::SimplexMesh & mesh)
{
    std::pair<Eigen::VectorXi, Eigen::MatrixXd> IC = mesh.first_point_collision( p );
    const int cell_ind = IC.first(0);
    const Eigen::VectorXd affine_coords = IC.second.col(0);
    const int dim = affine_coords.size() - 1;

    double          vol = 0.0;

    Eigen::VectorXd mu(dim);
    mu.setZero();

    Eigen::MatrixXd Sigma(dim,dim);
    Sigma.setZero();

    double tau = 0.0

    for ( int kk=0; kk<dim+1; ++kk )
    {
        const Ellipsoid & Ek = ellipsoids_at_mesh_nodes[mesh.cells(kk, cell_ind)];
        vol   += affine_coords(kk) * Ek.vol;
        mu    += affine_coords(kk) * Ek.mu;
        Sigma += affine_coords(kk) * Ek.Sigma;
        tau   += affine_coords(kk) * Ek.tau;
    }
    return Ellipsoid { vol, mu, Sigma, tau };
}


struct ImpulseResponseBatchData
{
    std::vector<Eigen::VectorXd> sample_points; // sample_points[j] has shape=(d,)

    vector<int> (*get_nearest_sample_point_inds)(const Eigen::VectorXd, int); // find indices of k-nearest sample points to query point
    bool        (*check_if_point_is_in_mesh)    (const Eigen::VectorXd);

    std::vector<double (*)(const Eigen::VectorXd)> psi_batches; // psi_batches[j] : shape=(d,) -> scalar
    std::vector<int> point2batch; // psi_batches[point2batch[j]] contains j'th impulse response
};


double eval_psi_j(const Eigen::VectorXd & p, const int j, const ImpulseResponseBatchData & IRBD, const SMESH::SimplexMesh & mesh)
{
    std::pair<Eigen::VectorXi, Eigen::MatrixXd> IC = mesh.first_point_collision( p );
    const int cell_ind = IC.first(0);
    const Eigen::VectorXd affine_coords = IC.second.col(0);
    const int dim = affine_coords.size() - 1;

    const int b = IRBD.point2batch[j];
    const Eigen::VectorXd & phi_j = IRBD.psi_batches[b];
    double psi_j_at_p = 0.0;
    for ( int kk=0; kk<dim+1; ++kk )
    {
        psi_j_at_p += affine_coords(kk) * phi_j(mesh.cells(kk, cell_ind)); // A
    }
    return psi_j_at_p;
}

VectorXi get_nearest_sample_point_inds(const Eigen::VectorXd & p, const int k, const ImpulseResponseBatchData & IRBD)
{
    const int k_prime = min(k, IRBD.sample_points.size())
    return IRBD.kdtree.query( x, k_prime ).first;
}

bool check_if_point_is_in_mesh(const Eigen::VectorXd & p, const SMESH::SimplexMesh & mesh)
{
    std::pair<Eigen::VectorXi, Eigen::MatrixXd> IC = mesh.first_point_collision( p );
    return (IC.first(0) >= 0);
}

std::vector<std::pair<Eigen::VectorXd, double>> interpolation_points_and_values(const Eigen::VectorXd &          y, // shape=(d,)
                                                                                const Eigen::VectorXd &          x, // shape=(d,)
                                                                                const int                        num_neighbors,
                                                                                const ImpulseResponseBatchData & IRBD,
                                                                                const EllipsoidData &            ED,
                                                                                const SMESH::SimplexMesh &       mesh)
{
    const Ellipsoid Ex = get_ellipsoid_at_point(x, ED, mesh);

    const VectorXi nearest_inds = get_nearest_sample_point_inds(x, num_neighbors, IRBD);
    const int N_nearest = nearest_inds.size();

    std::vector<std::pair<Eigen::VectorXd, double>> good_points_and_values;
    good_points_and_values.reserve(N_nearest);
    for ( int jj=0; jj<N_nearest; ++jj )
    {
        const int xj_ind = nearest_inds[jj];
        const Eigen::VectorXd xj = IRBD.sample_points[xj_ind];
        const Eigen::VectorXd z = y - x + xj;
        if ( check_if_point_is_in_mesh(z, mesh) )
        {
            const Ellipsoid Exj = get_ellipsoid_at_point(xj, ED, mesh);

            double varphi_j_at_point = 0.0;
            if ( point_is_in_ellipsoid(z, Exj) ) // if point is in the ellipsoid
            {
                varphi_j_at_point = Exj.vol * eval_psi_j(z, nearest_inds[jj], IRBD);
            }
            good_points_and_values.push_back(make_pair(xj - x, varphi_j_at_point));
        }
    }
    return good_points_and_values;
}


class ProductConvolutionKernelRBF : public HLIB::TCoeffFn< real_t >
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
                                 std::vector<Eigen::VectorXd>       col_coords,
                                 std::vector<Eigen::VectorXd>       row_coords,
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
        // Check input sizes
        bool input_is_good = true;
        for ( int rr : rowidxs )
        {
            if ( rr < 0 )
            {
                std::string error_message = "Negative row index. rr=";
                error_message += std::to_string(rr);
                throw std::invalid_argument( error_message );
            }
            else if ( rr >= row_coords.size() )
            {
                std::string error_message = "Row index too big. rr=";
                error_message += std::to_string(rr);
                error_message += ", row_coords.size()=";
                error_message += std::to_string(row_coords.size());
                throw std::invalid_argument( error_message );
            }
        }
        for ( int cc : colidxs )
        {
            if ( cc < 0 )
            {
                std::string error_message = "Negative col index. cc=";
                error_message += std::to_string(cc);
                throw std::invalid_argument( error_message );
            }
            else if ( cc >= col_coords.size() )
            {
                std::string error_message = "Col index too big. cc=";
                error_message += std::to_string(cc);
                error_message += ", col_coords.size()=";
                error_message += std::to_string(col_coords.size());
                throw std::invalid_argument( error_message );
            }
        }

        int nrow = rowidxs.size();
        int ncol = colidxs.size();
        for ( size_t  jj = 0; jj < ncol; ++jj )
        {
            for ( size_t  ii = 0; ii < nrow; ++ii )
            {
                matrix[ jj*nrow + ii ] = eval_matrix_entry(rowidxs[ii], colidxs[jj]);
                matrix[ jj*nrow + ii ] += 1.0e-14; // Code segfaults without this
            }
        }

    }

    using HLIB::TCoeffFn< real_t >::eval;

    virtual matform_t  matrix_format  () const { return MATFORM_NONSYM; }

};

} // end namespace PCK
