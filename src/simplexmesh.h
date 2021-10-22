#pragma once

#include <iostream>
#include <list>

#include <math.h>
#include <Eigen/Dense>

#include "kdtree.h"
#include "aabbtree.h"

using namespace Eigen;
using namespace std;


inline VectorXd projected_affine_coordinates( const VectorXd & query,  // shape=(dim, 1)
                                              const MatrixXd & points ) // shape=(dim, npts)
{
    int npts = points.cols();
    VectorXd coords(npts);

    if ( npts == 1 )
    {
        coords(0) = 1.0;
    }
    else if ( npts == 2 )
    {
        const VectorXd dv = points.col(1) - points.col(0);
        coords(1) = dv.dot(query - points.col(0)) / dv.squaredNorm();
        coords(0) = 1.0 - coords(1);
    }
    else
    {
        const MatrixXd dV = points.rightCols(npts-1).colwise() - points.col(0);
//        double x = dV.colPivHouseholderQr();
        coords.tail(npts-1) = dV.colPivHouseholderQr().solve(query - points.col(0)); // min_c ||dV*c - b||^2
//        coords.tail(npts-1) = dV.householderQr().solve(query - points.col(0)); // min_c ||dV*c - b||^2
        coords(0) = 1.0 - coords.tail(npts-1).sum();
    }
    return coords;
}


int power_of_two( int k ) // x=2^k, with 2^0 = 1. Why does c++ not have this!?
{
    int x = 1;
    for ( int ii=1; ii<=k; ++ii )
    {
        x = 2*x;
    }
    return x;
}

inline MatrixXd select_columns( const MatrixXd & A,    // shape=(N,M)
                                const VectorXi & inds) // shape=(k,1)
{
    int N = A.rows();
    int k = inds.size();
    MatrixXd A_selected;
    A_selected.resize(N, k);
    for ( int ii=0; ii<k; ++ii)
    {
        A_selected.col(ii) = A.col(inds(ii));
    }
    return A_selected;
}

inline MatrixXd select_columns( const MatrixXd &                 A,           // shape=(N,M)
                                const Matrix<bool, Dynamic, 1> & is_selected) // shape=(M,1)
{
    int N = A.rows();
    int M = A.cols();
    int K = is_selected.count();
    MatrixXd A_selected;
    A_selected.resize(N, K);
    int kk = 0;
    for ( int ii=0; ii<M; ++ii)
    {
        if ( is_selected(ii) )
        {
            A_selected.col(kk) = A.col(ii);
            kk = kk + 1;
        }
    }
    return A_selected;
}

inline VectorXd closest_point_in_simplex( const VectorXd & query,             // shape=(dim, 1)
                                          const MatrixXd & simplex_vertices ) // shape=(dim, npts)
{
    int dim = simplex_vertices.rows();
    int npts = simplex_vertices.cols();
    VectorXd closest_point(dim);

    if ( npts == 1 )
    {
        closest_point = simplex_vertices.col(0);
    }
    else
    {
        int num_facets = power_of_two(npts) - 1;
        Matrix<bool, Dynamic, Dynamic, RowMajor> all_facet_inds;
        all_facet_inds.resize(num_facets, npts);

        if ( npts == 2 )
        {
            all_facet_inds << false, true,
                              true,  false,
                              true,  true;
        }
        else if ( npts == 3 )
        {
            all_facet_inds << false, false, true,
                              false, true,  false,
                              false, true,  true,
                              true,  false, false,
                              true,  false, true,
                              true,  true,  false,
                              true,  true,  true;
        }
        else if ( npts == 4 )
        {
            all_facet_inds << false, false, false, true,
                              false, false, true,  false,
                              false, false, true,  true,
                              false, true,  false, false,
                              false, true,  false, true,
                              false, true,  true,  false,
                              false, true,  true,  true,
                              true,  false, false, false,
                              true,  false, false, true,
                              true,  false, true,  false,
                              true,  false, true,  true,
                              true,  true,  false, false,
                              true,  true,  false, true,
                              true,  true,  true,  false,
                              true,  true,  true,  true;
        }
        else
        {
            cout << "not implemented for npts>4."
                 << "Also, algorithm not recommended for large npts since it scales combinatorially." << endl;
        }

        closest_point = simplex_vertices.col(0);
        double dsq_best = (closest_point - query).squaredNorm();
        for ( int ii=0; ii<num_facets; ++ii ) // for each facet
        {
            Matrix<bool, Dynamic, 1> facet_inds = all_facet_inds.row(ii);
            MatrixXd facet_vertices = select_columns( simplex_vertices, facet_inds );
            VectorXd facet_coords( facet_vertices.cols() );
            facet_coords = projected_affine_coordinates( query, facet_vertices );
            bool projection_is_in_facet = (facet_coords.array() >= 0.0).all();
            if ( projection_is_in_facet )
            {
                VectorXd candidate_point = facet_vertices * facet_coords;
                double dsq_candidate = (candidate_point - query).squaredNorm();
                if ( dsq_candidate < dsq_best )
                {
                    closest_point = candidate_point;
                    dsq_best = dsq_candidate;
                }
            }
        }
    }
    return closest_point;
}

MatrixXd closest_point_in_simplex_vectorized( const MatrixXd & query,            // shape=(dim, nquery)
                                              const MatrixXd & simplex_vertices) // shape=(dim*npts, nquery)
{
    int dim = query.rows();
    int nquery = query.cols();
    int npts = simplex_vertices.rows() / dim;

    MatrixXd closest_point;
    closest_point.resize(query.rows(), query.cols());

    for ( int ii=0; ii<nquery; ++ii )
    {
        MatrixXd S;
        S.resize(dim, npts);
        for ( int jj=0; jj<dim; ++jj )
        {
            for ( int kk=0; kk<npts; ++kk )
            {
                S(jj,kk) = simplex_vertices(kk*dim + jj, ii);
            }
        }
        closest_point.col(ii) = closest_point_in_simplex( query.col(ii), S );
    }
    return closest_point;
}

template <int K>
class SimplexMesh
{
private:
    typedef Matrix<double, K, 1> KDVector;

    Matrix<double, K,   Dynamic> vertices;
    Matrix<int,    K+1, Dynamic> cells;
    Matrix<double, K,   Dynamic> box_mins;
    Matrix<double, K,   Dynamic> box_maxes;
    KDTree<K> kdtree;
    AABBTree<K> aabbtree;
    vector< Matrix<double, K+1, K> > simplex_transform_matrices;
    vector< Matrix<double, K+1, 1> > simplex_transform_vectors;

    int num_vertices;
    int num_cells;

public:
    SimplexMesh( const Ref<const Matrix<double, K,   Dynamic>> input_vertices,
                 const Ref<const Matrix<int   , K+1, Dynamic>> input_cells )
    {
        vertices = input_vertices; // copy
        cells = input_cells; // copy

        num_vertices = input_vertices.cols();
        num_cells = input_cells.cols();

        // Compute box min and max points for each cell
        box_mins.resize(K, num_cells);
        box_maxes.resize(K, num_cells);
        for ( int cc=0; cc<num_cells; ++cc)
        {
            for ( int kk=0; kk<K; ++kk )
            {
                double min_k = vertices(kk, cells(0, cc));
                double max_k = vertices(kk, cells(0, cc));
                for ( int vv=1; vv<K+1; ++vv)
                {
                    double candidate_min_k = vertices(kk, cells(vv, cc));
                    double candidate_max_k = vertices(kk, cells(vv, cc));
                    if (candidate_min_k < min_k)
                    {
                        min_k = candidate_min_k;
                    }
                    if (candidate_max_k > max_k)
                    {
                        max_k = candidate_max_k;
                    }
                }
                box_mins(kk, cc) = min_k;
                box_maxes(kk, cc) = max_k;
            }
        }

        kdtree = KDTree<K>( vertices.transpose() );
        aabbtree = AABBTree<K>( box_mins.transpose(), box_maxes.transpose() );

        simplex_transform_matrices.resize(num_cells);
        simplex_transform_vectors.resize(num_cells);
        for ( int ii=0; ii<num_cells; ++ii )
        {
            Matrix<double, K, 1> v0 = vertices.col(cells(0, ii));
            Matrix<double, K, K> dV;
            for ( int jj=0; jj<K; ++jj )
            {
                dV.col(jj) = vertices.col(cells(jj+1, ii)) - v0;
            }
            Matrix<double, K+1, K> S;
            Matrix<double, K, K> S0 = dV.colPivHouseholderQr().solve(MatrixXd::Identity(K,K));
            Matrix<double, 1, K> ones_rowvec;
            ones_rowvec.setOnes();
            S.bottomRightCorner(K, K) = S0;
            S.row(0) = - ones_rowvec * S0;
            simplex_transform_matrices[ii] = S;


            Matrix<double, K+1, 1> e0;
            e0.setZero();
            e0(0) = 1.0;
            simplex_transform_vectors[ii] = e0 - S * v0;
        }

    }

    inline bool point_is_in_mesh( KDVector query )
    {
        return (index_of_first_simplex_containing_point( query ) >= 0);
    }

    Matrix<bool, Dynamic, 1> point_is_in_mesh_vectorized( const Ref<const Matrix<double, K, Dynamic>> query_points )
    {
        int nquery = query_points.cols();
        Matrix<bool, Dynamic, 1> in_mesh;
        in_mesh.resize(nquery, 1);
        for ( int ii=0; ii<nquery; ++ii )
        {
            in_mesh(ii) = point_is_in_mesh( query_points.col(ii) );
        }
        return in_mesh;
    }

    KDVector closest_point( KDVector query )
    {
        KDVector closest_point = vertices.col(0);
        if ( point_is_in_mesh( query ) )
        {
            closest_point = query;
        }
        else
        {
            pair<KDVector, double> kd_result = kdtree.nearest_neighbor( query );
            double dist_estimate = (1.0 + 1e-14) * sqrt(kd_result.second);
            vector<int> candidate_inds = aabbtree.all_ball_intersections( query, dist_estimate );
            int num_candidates = candidate_inds.size();

            double dsq_best = (closest_point - query).squaredNorm();
            for ( int ii=0; ii<num_candidates; ++ii )
            {
                int ind = candidate_inds[ii];
                Matrix<double, K, K+1> simplex_vertices;
                for ( int jj=0; jj<K+1; ++jj )
                {
                    simplex_vertices.col(jj) = vertices.col(cells(jj, ind));
                }

                KDVector candidate = closest_point_in_simplex( query, simplex_vertices );
                double dsq_candidate = (candidate - query).squaredNorm();
                if ( dsq_candidate < dsq_best )
                {
                    closest_point = candidate;
                    dsq_best = dsq_candidate;
                }
            }
        }
        return closest_point;
    }

    Matrix<double, K, Dynamic> closest_point_vectorized( const Ref<const Matrix<double, K, Dynamic>> query_points )
    {
        int num_queries = query_points.cols();
        Matrix<double, K, Dynamic> closest_points;
        closest_points.resize(K, num_queries);
        for ( int ii=0; ii<num_queries; ++ii )
        {
            closest_points.col(ii) = closest_point( query_points.col(ii) );
        }
        return closest_points;
    }

    inline VectorXd simplex_coordinates( int simplex_ind, const KDVector query )
    {
        return simplex_transform_matrices[simplex_ind] * query + simplex_transform_vectors[simplex_ind];
    }

    inline int index_of_first_simplex_containing_point( const KDVector query )
    {
        vector<int> candidate_inds =  aabbtree.all_point_intersections( query );
        int num_candidates = candidate_inds.size();
        int ind = -1;
        for ( int ii=0; ii<num_candidates; ++ii )
        {
            int candidate_ind = candidate_inds[ii];
            VectorXd affine_coords = simplex_coordinates( candidate_ind, query );
            bool point_is_in_simplex = (affine_coords.array() >= 0.0).all();
            if ( point_is_in_simplex )
            {
                ind = candidate_ind;
                break;
            }
        }
        return ind;
    }

    MatrixXd evaluate_functions_at_points( const Ref<const MatrixXd> functions_at_vertices, // shape=(num_functions, num_vertices)
                                           const Ref<const Matrix<double, K, Dynamic>> points ) // shape=(K, num_pts)
    {
        int num_functions = functions_at_vertices.rows();
        int num_pts = points.cols();

        MatrixXd function_at_points;
        function_at_points.resize(num_functions, num_pts);
        function_at_points.setZero();

        for ( int ii=0; ii<num_pts; ++ii ) // for each point
        {
            vector<int> candidate_inds =  aabbtree.all_point_intersections( points.col(ii) );
            int num_candidates = candidate_inds.size();
            for ( int jj=0; jj<num_candidates; ++jj ) // for each candidate simplex that the point might be in
            {
                int simplex_ind = candidate_inds[jj];
                Matrix<double, K+1, 1> affine_coords = simplex_transform_matrices[simplex_ind] * points.col(ii)
                                                       + simplex_transform_vectors[simplex_ind];
                bool point_is_in_simplex = (affine_coords.array() >= 0.0).all();
                if ( point_is_in_simplex )
                {
                    for ( int kk=0; kk<K+1; ++kk ) // for simplex vertex
                    {
                        for ( int ll=0; ll<num_functions; ++ll ) // for each function
                        {
                            function_at_points(ll, ii) += affine_coords(kk) * functions_at_vertices(ll, cells(kk, simplex_ind));
                        }
                    }
                    break;
                }
            }
        }
        return function_at_points;
    }

//    inline double evaluate_function_at_point( const Ref<const VectorXd> function_at_vertices,
//                                              const KDVector point )
//    {
//        vector<int> candidate_inds =  aabbtree.all_point_intersections( point );
//        double function_at_point = 0.0;
//        int num_candidates = candidate_inds.size();
//        for ( int ii=0; ii<num_candidates; ++ii )
//        {
//            int ind = candidate_inds[ii];
//            VectorXd affine_coords = simplex_coordinates( ind, point );
//            bool point_is_in_simplex = (affine_coords.array() >= 0.0).all();
//            if ( point_is_in_simplex )
//            {
//                for ( int vv=0; vv<K+1; ++vv)
//                {
//                    function_at_point += affine_coords(vv) * function_at_vertices(cells(vv, ind));
//                }
//                break;
//            }
//        }
//        return function_at_point;
//    }
//
////    VectorXd evaluate_function_at_point_vectorized( const VectorXd &          function_at_vertices,
////                                                    const Ref<const MatrixXd> points )
//    VectorXd evaluate_function_at_point_vectorized( const Ref<const VectorXd>                            function_at_vertices,
//                                                    const Ref<const Array<double, Dynamic, K, RowMajor>> points )
//    {
//        int npts = points.rows();
//        VectorXd function_at_points(npts);
//        for ( int ii=0; ii<npts; ++ii )
//        {
//            KDVector p = points.row(ii);
//            function_at_points(ii) = evaluate_function_at_point( function_at_vertices, p );
//        }
//        return function_at_points;
//    }
};


