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
    typedef Array<double, K, 1> KDVector;

    Array<double, Dynamic, K, RowMajor> vertices;
    Array<int, Dynamic, K+1, RowMajor> cells;
    Array<double, Dynamic, K, RowMajor> box_mins;
    Array<double, Dynamic, K, RowMajor> box_maxes;
    KDTree<K> kdtree;
    AABBTree<K> aabbtree;

public:
    SimplexMesh( const Ref<const Array<double, Dynamic, K>> input_vertices,
                 const Ref<const Array<int, Dynamic, K+1>>  input_cells )
    {
        // Copy input vertices into local array
        int num_vertices = input_vertices.rows();
        vertices.resize(num_vertices, K);
        for ( int ii=0; ii<num_vertices; ++ii )
        {
            vertices.row(ii) = input_vertices.row(ii);
        }

        // Copy input cells into local array
        int num_cells = input_cells.rows();
        cells.resize(num_cells, K+1);
        for ( int ii=0; ii<num_cells; ++ii)
        {
            cells.row(ii) = input_cells.row(ii);
        }

        // Compute box min and max points for each cell
        box_mins.resize(num_cells, K);
        box_maxes.resize(num_cells, K);
        for ( int cc=0; cc<num_cells; ++cc)
        {
            for ( int kk=0; kk<K; ++kk )
            {
                double min_k = vertices(cells(cc,0), kk);
                double max_k = vertices(cells(cc,0), kk);
                for ( int vv=1; vv<K+1; ++vv)
                {
                    double candidate_min_k = vertices(cells(cc,vv), kk);
                    double candidate_max_k = vertices(cells(cc,vv), kk);
                    if (candidate_min_k < min_k)
                    {
                        min_k = candidate_min_k;
                    }
                    if (candidate_max_k > max_k)
                    {
                        max_k = candidate_max_k;
                    }
                }
                box_mins(cc, kk) = min_k;
                box_maxes(cc, kk) = max_k;
            }
        }

        kdtree = KDTree<K>( vertices );
        aabbtree = AABBTree<K>( box_mins, box_maxes );
    }

    inline bool point_is_in_mesh( KDVector query )
    {
        return (index_of_first_simplex_containing_point( query ) >= 0);
    }

    Array<bool, Dynamic, 1> point_is_in_mesh_vectorized( Array<double, Dynamic, K> query_points )
    {
        int nquery = query_points.rows();
        Array<bool, Dynamic, 1> in_mesh;
        in_mesh.resize(nquery, 1);
        for ( int ii=0; ii<nquery; ++ii )
        {
            in_mesh(ii) = point_is_in_mesh( query_points.row(ii) );
        }
        return in_mesh;
    }

    KDVector closest_point( KDVector query )
    {
        KDVector closest_point = vertices.row(0);
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

    //        cout << "num_candidates=" << num_candidates << endl;

            double dsq_best = (closest_point - query).matrix().squaredNorm();
            for ( int ii=0; ii<num_candidates; ++ii )
            {
                int ind = candidate_inds[ii];
                Matrix<double, K, K+1> simplex_vertices;
                for ( int jj=0; jj<K+1; ++jj )
                {
                    simplex_vertices.col(jj) = vertices.row(cells(ind,jj));
                }

                KDVector candidate = closest_point_in_simplex( query, simplex_vertices );
                double dsq_candidate = (candidate - query).matrix().squaredNorm();
                if ( dsq_candidate < dsq_best )
                {
                    closest_point = candidate;
                    dsq_best = dsq_candidate;
                }
            }
        }
        return closest_point;
    }

    Array<double, Dynamic, K> closest_point_vectorized( Array<double, Dynamic, K> query_points )
    {
        int num_queries = query_points.rows();
        Array<double, Dynamic, K> closest_points;
        closest_points.resize(num_queries, K);
        for ( int ii=0; ii<num_queries; ++ii )
        {
            closest_points.row(ii) = closest_point( query_points.row(ii) );
        }
        return closest_points;
    }

    inline VectorXd affine_coordinates_of_point_in_simplex( int simplex_ind, KDVector query )
    {
        Matrix<double, K, K+1> S;
        for ( int jj=0; jj<K+1; ++jj )
        {
//            cout << "cells.rows()=" << cells.rows() << endl;
//            cout << "simplex_ind=" << simplex_ind << endl;
            S.col(jj) = vertices.row(cells(simplex_ind, jj));
//            S.col(jj) = vertices.row(cells(0, jj));
        }

        VectorXd affine_coords = projected_affine_coordinates( query, S );
        return affine_coords;
    }

    inline int index_of_first_simplex_containing_point( KDVector query )
    {
        vector<int> candidate_inds =  aabbtree.all_point_intersections( query );
        int num_candidates = candidate_inds.size();
        int ind = -1;
        for ( int ii=0; ii<num_candidates; ++ii )
        {
            int candidate_ind = candidate_inds[ii];
            VectorXd affine_coords = affine_coordinates_of_point_in_simplex( candidate_ind, query );
            bool point_is_in_simplex = (affine_coords.array() >= 0.0).all();
            if ( point_is_in_simplex )
            {
                ind = candidate_ind;
                break;
            }
        }
        return ind;
    }

    inline double evaluate_function_at_point( const VectorXd & function_at_vertices,
                                              const KDVector & point )
    {
        vector<int> candidate_inds =  aabbtree.all_point_intersections( point );
        double function_at_point = 0.0;
        int num_candidates = candidate_inds.size();
        for ( int ii=0; ii<num_candidates; ++ii )
        {
            int ind = candidate_inds[ii];
            VectorXd affine_coords = affine_coordinates_of_point_in_simplex( ind, point );
            bool point_is_in_simplex = (affine_coords.array() >= 0.0).all();
            if ( point_is_in_simplex )
            {
                for ( int vv=0; vv<K+1; ++vv)
                {
                    function_at_point += affine_coords(vv) * function_at_vertices(cells(ind, vv));
                }
                break;
            }
        }
        return function_at_point;
    }

    VectorXd evaluate_function_at_point_vectorized( const VectorXd &          function_at_vertices,
                                                    const Ref<const MatrixXd> points )
    {
        int npts = points.rows();
        VectorXd function_at_points(npts);
        for ( int ii=0; ii<npts; ++ii )
        {
            function_at_points(ii) = evaluate_function_at_point( function_at_vertices, points.row(ii) );
        }
        return function_at_points;
    }
};

