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


vector<vector<int>> powerset(int N)
{
    vector<vector<int>> pset;
    if (N < 1)
    {
        vector<int> empty_set;
        pset.push_back( empty_set );
    }
    else
    {
        pset = powerset(N-1);
        int sz0 = pset.size();
        for ( int ii=0; ii<sz0; ++ii )
        {
            vector<int> x = pset[ii];
            x.push_back(N-1);
            pset.push_back(x);
        }
    }
    return pset;
}

std::pair<MatrixXd, VectorXd> make_simplex_transform_stuff( const MatrixXd & simplex_vertices )
{
    int K = simplex_vertices.rows();
    int M = simplex_vertices.cols();

    MatrixXd S(M, K); // first return
    VectorXd b(M);    // second return

    if ( M == 1 )
    {
        S.setZero();
        b.setOnes();
    }
    else
    {
        VectorXd v0(K);
        v0 = simplex_vertices.col(0);
        MatrixXd dV(K,M-1);
        for ( int jj=1; jj<M; ++jj )
        {
            dV.col(jj-1) = simplex_vertices.col(jj) - v0;
        }
        MatrixXd S0(M-1, K);
        S0 = dV.colPivHouseholderQr().solve(MatrixXd::Identity(K,K));
        Matrix<double, 1, Dynamic> ones_rowvec(M-1);
        ones_rowvec.setOnes();
        S.bottomRightCorner(M-1, K) = S0;
        S.row(0) = -ones_rowvec * S0;

        VectorXd e0(M);
        e0.setZero();
        e0(0) = 1.0;
        b = e0 - S * v0;
    }
    return std::make_pair(S, b);
}

struct FacetStuff { vector<MatrixXd> VV;   // Facet vertices
                    vector<MatrixXd> SS;   // Facet simplex coordinate matrices
                    vector<VectorXd> bb;}; // Facet simplex coordinate vectors

FacetStuff make_facet_stuff( const MatrixXd & simplex_vertices )
{
    // q = query point
    // pi = projection of q onto ith facet
    // ci = affine coordinates of pi with respect to vertices of ith facet
    // ci = Si * q + bi
    // pi = Vi * ci
    int K = simplex_vertices.rows();
    int num_vertices = simplex_vertices.cols();

    vector<VectorXd> all_vertices(num_vertices);
    for ( int ii=0; ii<num_vertices; ++ii)
    {
        all_vertices[ii] = simplex_vertices.col(ii);
    }

    vector<vector<int>> pset = powerset(num_vertices);

    vector<MatrixXd> VV; // [V1, V2, ...]
    vector<MatrixXd> SS; // [S1, S2, ...]
    vector<VectorXd> bb; // [b1, b2, ...]
    for ( int ii=0; ii<pset.size(); ++ii )
    {
        vector<int> facet_inds = pset[ii];
        if ( !facet_inds.empty() )
        {
            int num_facet_vertices = facet_inds.size();
            MatrixXd Vi(K, num_facet_vertices);
            for ( int jj=0; jj<num_facet_vertices; ++jj )
            {
                Vi.col(jj) = simplex_vertices.col(facet_inds[jj]);
            }

            std::pair<MatrixXd, VectorXd> res = make_simplex_transform_stuff( Vi );

            VV.push_back(Vi);
            SS.push_back(res.first);
            bb.push_back(res.second);
        }
    }
    return FacetStuff { VV, SS, bb };
}

VectorXd closest_point_in_simplex_using_precomputed_facet_stuff( const VectorXd & query,            // shape=(dim, 1)
                                                                 const FacetStuff & FS)
{
    int dim = query.size();
    VectorXd closest_point(dim);

    int num_facets = FS.VV.size();

    double dsq_best = std::numeric_limits<double>::infinity();
    for ( int ii=0; ii<num_facets; ++ii )
    {
        VectorXd facet_coords = FS.SS[ii] * query + FS.bb[ii];
        bool projection_is_in_facet = (facet_coords.array() >= 0.0).all();
        if ( projection_is_in_facet )
        {
            VectorXd candidate_point = FS.VV[ii] * facet_coords;
            double dsq_candidate = (candidate_point - query).squaredNorm();
            if ( dsq_candidate < dsq_best )
            {
                closest_point = candidate_point;
                dsq_best = dsq_candidate;
            }
        }
    }


//
//    if ( npts == 1 )
//    {
//        closest_point = simplex_vertices.col(0);
//    }
//    else
//    {
//        int num_facets = power_of_two(npts) - 1;
//        Matrix<bool, Dynamic, Dynamic, RowMajor> all_facet_inds;
//        all_facet_inds.resize(num_facets, npts);
//
//        if ( npts == 2 )
//        {
//            all_facet_inds << false, true,
//                              true,  false,
//                              true,  true;
//        }
//        else if ( npts == 3 )
//        {
//            all_facet_inds << false, false, true,
//                              false, true,  false,
//                              false, true,  true,
//                              true,  false, false,
//                              true,  false, true,
//                              true,  true,  false,
//                              true,  true,  true;
//        }
//        else if ( npts == 4 )
//        {
//            all_facet_inds << false, false, false, true,
//                              false, false, true,  false,
//                              false, false, true,  true,
//                              false, true,  false, false,
//                              false, true,  false, true,
//                              false, true,  true,  false,
//                              false, true,  true,  true,
//                              true,  false, false, false,
//                              true,  false, false, true,
//                              true,  false, true,  false,
//                              true,  false, true,  true,
//                              true,  true,  false, false,
//                              true,  true,  false, true,
//                              true,  true,  true,  false,
//                              true,  true,  true,  true;
//        }
//        else
//        {
//            cout << "not implemented for npts>4."
//                 << "Also, algorithm not recommended for large npts since it scales combinatorially." << endl;
//        }
//
//        closest_point = simplex_vertices.col(0);
//        double dsq_best = (closest_point - query).squaredNorm();
//        for ( int ii=0; ii<num_facets; ++ii ) // for each facet
//        {
//            Matrix<bool, Dynamic, 1> facet_inds = all_facet_inds.row(ii);
//            MatrixXd facet_vertices = select_columns( simplex_vertices, facet_inds );
//            VectorXd facet_coords( facet_vertices.cols() );
//            facet_coords = projected_affine_coordinates( query, facet_vertices );
//            bool projection_is_in_facet = (facet_coords.array() >= 0.0).all();
//            if ( projection_is_in_facet )
//            {
//                VectorXd candidate_point = facet_vertices * facet_coords;
//                double dsq_candidate = (candidate_point - query).squaredNorm();
//                if ( dsq_candidate < dsq_best )
//                {
//                    closest_point = candidate_point;
//                    dsq_best = dsq_candidate;
//                }
//            }
//        }
//    }
    return closest_point;
}

VectorXd closest_point_in_simplex( const VectorXd & query,            // shape=(dim, 1)
                                   const MatrixXd & simplex_vertices) // shape=(dim, npts)
{
    FacetStuff FS = make_facet_stuff( simplex_vertices );
    return closest_point_in_simplex_using_precomputed_facet_stuff(query, FS);
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
    vector< FacetStuff > all_facet_stuff;

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
        aabbtree = AABBTree<K>( box_mins, box_maxes );

        simplex_transform_matrices.resize(num_cells);
        simplex_transform_vectors.resize(num_cells);
        all_facet_stuff.resize(num_cells);
        for ( int ii=0; ii<num_cells; ++ii )
        {
            Matrix<double, K, K+1> simplex_vertices;
            for (int jj=0; jj<K+1; ++jj )
            {
                simplex_vertices.col(jj) = vertices.col(cells(jj, ii));
            }
            std::pair<MatrixXd, VectorXd> res = make_simplex_transform_stuff( simplex_vertices );
            simplex_transform_matrices[ii] = res.first;
            simplex_transform_vectors[ii] = res.second;

            all_facet_stuff[ii] = make_facet_stuff( simplex_vertices );
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

    KDVector closest_point( const KDVector & query )
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

                KDVector candidate = closest_point_in_simplex_using_precomputed_facet_stuff( query, all_facet_stuff[ind] );
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


