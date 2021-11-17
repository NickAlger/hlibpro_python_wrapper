#pragma once

#include <iostream>
#include <list>
#include <stdexcept>

//#include <thread>
//#include <execution>
//#include <chrono>

#include "thread-pool-master/thread_pool.hpp"

#include <math.h>
#include <Eigen/Dense>

#include "kdtree.h"
#include "aabbtree.h"
//#include "geometric_sort.h"


namespace SMESH {

using namespace Eigen;
using namespace std;
using namespace KDT;
using namespace AABB;


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


//int power_of_two( int k ) // x=2^k, with 2^0 = 1. Why does c++ not have this!?
//{
//    int x = 1;
//    for ( int ii=1; ii<=k; ++ii )
//    {
//        x = 2*x;
//    }
//    return x;
//}

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

std::pair<MatrixXd, VectorXd> make_simplex_transform_operator( const MatrixXd & simplex_vertices )
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

struct Simplex { MatrixXd V;   // simplex vertices
                 MatrixXd A;   // coordinate transform matrix
                 VectorXd b;}; // coordinate transform vector

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

            std::pair<MatrixXd, VectorXd> res = make_simplex_transform_operator( Vi );

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

pair<VectorXd, VectorXd> compute_pointcloud_bounding_box( const MatrixXd & points )
{
    int dim = points.rows();
    int num_points = points.cols();

    VectorXd box_min(dim);
    VectorXd box_max(dim);

    for ( int kk=0; kk<dim; ++kk )
    {
        double min_k = points(kk, 0);
        double max_k = points(kk, 0);
        for ( int vv=1; vv<num_points; ++vv)
        {
            double candidate_k = points(kk, vv);
            if (candidate_k < min_k)
            {
                min_k = candidate_k;
            }
            if (candidate_k > max_k)
            {
                max_k = candidate_k;
            }
        }
        box_min(kk) = min_k;
        box_max(kk) = max_k;
    }
    return std::make_pair(box_min, box_max);
}

template <int K>
struct ind_and_coords { int simplex_ind;
                        Matrix<double, K+1, 1> affine_coords; };

template <int K>
class SimplexMesh
{
private:
    typedef Matrix<double, K, 1> KDVector;

    Matrix<int,    K,   Dynamic> faces;    // boundary face simplices of dim K-1
    vector<VectorXi>             subfaces; // sub-simplices of boundary faces of dim 0 through K-1 (includes boundary faces)

    vector<VectorXi> face2subface;

    AABBTree cell_aabbtree;
    AABBTree face_aabbtree;
    KDTree   face_kdtree;

    vector< Simplex > cell_simplices;
    vector< Simplex > subface_simplices;

    thread_pool pool;

    int num_vertices;
    int num_cells;
    int num_faces;
    int num_subfaces;

    int default_sleep_duration;
    int default_number_of_threads;

public:
    Matrix<double, K,   Dynamic> vertices;
    Matrix<int,    K+1, Dynamic> cells;    // interior simplices of dim K

    SimplexMesh( const Ref<const Matrix<double, K,   Dynamic>> input_vertices,
                 const Ref<const Matrix<int   , K+1, Dynamic>> input_cells )
    {
        // ------------------------    Input checking and copying    ------------------------
        num_vertices = input_vertices.cols();
        num_cells = input_cells.cols();

        if ( num_vertices < 1 )
        {
            throw std::invalid_argument( "no vertices provided" );
        }

        if ( num_cells < 1 )
        {
            throw std::invalid_argument( "no cells provided" );
        }

        if ( input_cells.minCoeff() < 0 )
        {
            throw std::invalid_argument( "at least one vertex index in input_cells is negative" );
        }

        if ( input_cells.maxCoeff() >= num_vertices )
        {
            throw std::invalid_argument( "at least one vertex index in input_cells >= num_vertices" );
        }

        vertices = input_vertices; // copy
        cells    = input_cells;    // copy


        // ------------------------    Multithreading stuff    ------------------------
        default_sleep_duration = pool.sleep_duration;
        default_number_of_threads = pool.get_thread_count();


        // ------------------------    CELLS    ------------------------
        // Generate cell simplices and transform operators
        cell_simplices.resize(num_cells);
        for ( int ii=0; ii<num_cells; ++ii )
        {
            Matrix<double, K, K+1> simplex_vertices;
            for (int jj=0; jj<K+1; ++jj )
            {
                simplex_vertices.col(jj) = vertices.col(cells(jj, ii));
            }
            std::pair<MatrixXd, VectorXd> STS = make_simplex_transform_operator( simplex_vertices );
            cell_simplices[ii] = Simplex { simplex_vertices, // V
                                           STS.first,        // A
                                           STS.second };     // b
        }

        // Generate cell AABB tree
        Matrix<double, K,   Dynamic> cell_box_mins(K, num_cells);
        Matrix<double, K,   Dynamic> cell_box_maxes(K, num_cells);
        for ( int ii=0; ii<num_cells; ++ii )
        {
            pair<VectorXd, VectorXd> BB = compute_pointcloud_bounding_box( cell_simplices[ii].V );
            cell_box_mins.col(ii) = BB.first;
            cell_box_maxes.col(ii) = BB.second;
        }
//        cell_aabbtree = AABBTree( cell_box_mins, cell_box_maxes );
        cell_aabbtree.build_tree( cell_box_mins, cell_box_maxes );


        // ------------------------    FACES    ------------------------
        // For all K-facets (K-1 dim simplex which has K vertices), compute how many cells they are part of.
        map<vector<int>, int> Kfacet_counts; // Kfacet -> cell count
        for ( int cc=0; cc<num_cells; ++cc )
        {
            for ( int opposite_vertex_ind=0; opposite_vertex_ind<K+1; ++opposite_vertex_ind )
            {
                vector<int> face;
                for ( int kk=0; kk<K+1; ++kk )
                {
                    if ( kk != opposite_vertex_ind )
                    {
                        face.push_back(cells(kk, cc));
                    }
                }
                sort( face.begin(), face.end() ); // sort for comparison purposes

                if ( Kfacet_counts.find(face) == Kfacet_counts.end() ) // if this face isnt in the map yet
                {
                    Kfacet_counts[face] = 1;
                }
                else
                {
                    Kfacet_counts[face] += 1;
                }
            }
        }

        // Faces (K-facets on the boundary) are the K-facets that are part of only one K+1 dim cell
        vector<Matrix<int, K, 1>> faces_vector;
        for ( auto it = Kfacet_counts.begin(); it != Kfacet_counts.end(); ++it )
        {
            vector<int> face = it->first;
            Matrix<int, K, 1> F;
            for ( int kk=0; kk<K; ++kk)
            {
                F(kk) = face[kk];
            }
            int count = it->second;
            if ( count == 1 )
            {
                faces_vector.push_back(F);
            }
        }

        num_faces = faces_vector.size();
        faces.resize(K, num_faces);
        for ( int ii=0; ii<num_faces; ++ii )
        {
            faces.col(ii) = faces_vector[ii];
        }


        // Create kdtree of vertices that are on faces (i.e., on the boundary)
        set<int> face_vertex_inds;
        for ( int bb=0; bb<num_faces; ++bb )
        {
            for ( int kk=0; kk<K; ++kk )
            {
                face_vertex_inds.insert(faces(kk, bb));
            }
        }

        int num_face_vertices = face_vertex_inds.size();
        MatrixXd face_vertices(K,num_face_vertices);
        int vv=0;
        for ( auto it  = face_vertex_inds.begin();
                   it != face_vertex_inds.end();
                 ++it )
        {
            face_vertices.col(vv) = vertices.col( *it );
            vv += 1;
        }
//        face_kdtree = KDTree( face_vertices );
        face_kdtree.build_tree( face_vertices );


        // Create face AABB tree
        Matrix<double, K,   Dynamic> face_box_mins(K, num_faces);
        Matrix<double, K,   Dynamic> face_box_maxes(K, num_faces);
        for ( int bb=0; bb<num_faces; ++bb )
        {
            Matrix<double, K, K> face_vertices;
            for (int jj=0; jj<K; ++jj )
            {
                face_vertices.col(jj) = vertices.col(faces(jj, bb));
            }
            pair<VectorXd, VectorXd> BB = compute_pointcloud_bounding_box( face_vertices );
            face_box_mins.col(bb) = BB.first;
            face_box_maxes.col(bb) = BB.second;
        }
//        face_aabbtree = AABBTree( face_box_mins, face_box_maxes );
        face_aabbtree.build_tree( face_box_mins, face_box_maxes );


        // ------------------------    SUBFACES    ------------------------
        // Construct all boundary entities (faces-of-faces, etc)
        vector<vector<int>> pset = powerset(K); // powerset(3) = [[], [0], [1], [0, 1], [2], [0, 2], [1, 2], [0, 1, 2]]
        map<vector<int>, vector<int>> subface2face_map;
        for ( int bb=0; bb<num_faces; ++bb )
        {
            for ( int ii=0; ii<pset.size(); ++ii )
            {
                vector<int> vertex_subset = pset[ii];
                if ( !vertex_subset.empty() )
                {
                    int num_subface_vertices = vertex_subset.size();
                    vector<int> subface_vertex_inds;
                    for ( int jj=0; jj<num_subface_vertices; ++jj )
                    {
                        subface_vertex_inds.push_back(faces(vertex_subset[jj], bb));
                    }
                    sort( subface_vertex_inds.begin(), subface_vertex_inds.end() );


                    if ( subface2face_map.find(subface_vertex_inds) == subface2face_map.end() ) // if this subface isnt in the map yet
                    {
                        vector<int> faces_containing_this_subface;
                        faces_containing_this_subface.push_back(bb);
                        subface2face_map[subface_vertex_inds] = faces_containing_this_subface;
                    }
                    else
                    {
                        subface2face_map[subface_vertex_inds].push_back(bb);
                    }
                }
            }
        }

        vector<vector<int>> face2subface_vector( num_faces );
        int subface_number = 0;
        for ( auto it  = subface2face_map.begin();
                   it != subface2face_map.end();
                 ++it )
        {
            vector<int> subface = it->first;
            int num_subface_vertices = subface.size();
            VectorXi SF(num_subface_vertices);
            for ( int kk=0; kk<num_subface_vertices; ++kk)
            {
                SF(kk) = subface[kk];
            }
            subfaces.push_back(SF);

            vector<int> faces_containing_this_subface = it->second;
            for ( int bb=0; bb<faces_containing_this_subface.size(); ++bb )
            {
                int face_ind = faces_containing_this_subface[bb];
                face2subface_vector[face_ind].push_back(subface_number);
            }
            subface_number += 1;
        }

        face2subface.resize(num_faces);
        for ( int bb=0; bb<num_faces; ++bb )
        {
            vector<int> entities_for_this_face_vector = face2subface_vector[bb];
            VectorXi entities_for_this_face(entities_for_this_face_vector.size());
            for ( int jj=0; jj<entities_for_this_face_vector.size(); ++jj )
            {
                entities_for_this_face(jj) = entities_for_this_face_vector[jj];
            }
            face2subface[bb] = entities_for_this_face;
        }

        num_subfaces = subfaces.size();

        subface_simplices.resize(num_subfaces);
        for ( int ee=0; ee<num_subfaces; ++ee )
        {
            int num_vertices_in_subface = subfaces[ee].size();
            MatrixXd subface_vertices(K, num_vertices_in_subface);
            for (int vv=0; vv<num_vertices_in_subface; ++vv )
            {
                subface_vertices.col(vv) = vertices.col(subfaces[ee](vv));
            }
            std::pair<MatrixXd, VectorXd> STS = make_simplex_transform_operator( subface_vertices );
            subface_simplices[ee] = Simplex { subface_vertices, // V
                                              STS.first,       // A
                                              STS.second };    // b
        }
    }

    inline bool point_is_in_mesh( KDVector query ) const
    {
        return (index_of_first_simplex_containing_point( query ) >= 0);
    }

    Matrix<bool, Dynamic, 1> point_is_in_mesh_vectorized( const Ref<const Matrix<double, K, Dynamic>> query_points ) const
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

    KDVector closest_point( const KDVector & query ) const
    {
        KDVector closest_point = vertices.col(0);
        if ( point_is_in_mesh( query ) )
        {
            closest_point = query;
        }
        else
        {
            // 1. Find a set of candidate boundary faces, one of which contains the closest point
            pair<VectorXi, VectorXd> kd_result = face_kdtree.query( query, 1 );
            double dist_estimate = (1.0 + 1e-14) * sqrt(kd_result.second(0));
            VectorXi face_inds = face_aabbtree.ball_collisions( query, dist_estimate );

            // 2. Determine unique set of boundary entities to visit
            vector<int> entities;
            entities.reserve(power_of_two(K));
            for ( int ii=0; ii<face_inds.size(); ++ii )
            {
                const VectorXi & subface_inds = face2subface[face_inds(ii)];
                for ( int jj=0; jj<subface_inds.size(); ++jj )
                {
                    entities.push_back(subface_inds(jj));
                }
            }
            sort( entities.begin(), entities.end() );
            entities.erase( unique( entities.begin(), entities.end() ), entities.end() );

            // 3. Project query onto the affine subspaces associated with each subface.
            // 4. Discard "bad" projections that to not land in their subface.
            // 5. Return closest "good" projection.
            double dsq_best = (closest_point - query).squaredNorm();
            for ( int ee=0; ee<entities.size(); ++ee )
            {
                const Simplex & E = subface_simplices[entities[ee]];
                VectorXd projected_affine_coords = E.A * query + E.b;
                if ( (projected_affine_coords.array() >= 0.0).all() ) // projection is in subface simplex
                {
                    KDVector projected_query = E.V * projected_affine_coords;
                    double dsq = (projected_query - query).squaredNorm();
                    if ( dsq < dsq_best )
                    {
                        closest_point = projected_query;
                        dsq_best = dsq;
                    }
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

        auto loop = [&](const int &a, const int &b)
        {
            for ( int ii=a; ii<b; ++ii )
            {
                closest_points.col(ii) = closest_point( query_points.col(ii) );
            }
        };

        pool.parallelize_loop(0, num_queries, loop);
        return closest_points;
    }

    inline VectorXd simplex_coordinates( int simplex_ind, const KDVector query ) const
    {
          const Simplex & S = cell_simplices[simplex_ind];
          return S.A * query + S.b;
    }

    inline int index_of_first_simplex_containing_point( const KDVector query ) const
    {
        VectorXi candidate_inds =  cell_aabbtree.point_collisions( query );
        int num_candidates = candidate_inds.size();
        int ind = -1;
        for ( int ii=0; ii<num_candidates; ++ii )
        {
            int candidate_ind = candidate_inds(ii);
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

    void get_simplex_ind_and_affine_coordinates_of_point( const KDVector & point, ind_and_coords<K> & IC ) const
    {
        IC.simplex_ind = -1;

        VectorXi candidate_inds =  cell_aabbtree.point_collisions( point );
        int num_candidates = candidate_inds.size();

        for ( int jj=0; jj<num_candidates; ++jj ) // for each candidate simplex that the point might be in
        {
            int candidate_simplex_ind = candidate_inds(jj);
            const Simplex & S = cell_simplices[candidate_simplex_ind];
            IC.affine_coords = S.A * point + S.b;
            if ( (IC.affine_coords.array() >= 0.0).all() ) // point is in simplex
            {
                IC.simplex_ind = candidate_simplex_ind;
                break;
            }
        }
    }


    // ------------    SimplexMesh::evaluate_functions_at_points()    --------------
    // INPUT:
    //   Finite element function nodal values:
    //      functions_at_vertices = [[f_1, f_2, f_3, f_4, ..., f_N],
    //                               [g_1, g_2, g_3, g_4, ..., g_N],
    //                               [h_1, h_2, h_3, h_4, ..., h_N]]
    //      - shape = (num_functions, num_vertices)
    //      - f(x) = sum_{i=1}^N f_i phi_i(x)
    //      - g(x) = sum_{i=1}^N g_i phi_i(x)
    //      - h(x) = sum_{i=1}^N h_i phi_i(x)
    //      - phi_i is CG1 FEM basis function (hat function)
    //
    //   Points to evaluate finite element functions at:
    //      points = [[p1_x, p2_x, p3_x, ..., pM_x],
    //                [p1_y, p2_y, p3_y, ..., pM_y]]
    //      - shape = (K, num_pts)
    //      - K = spatial dimension
    //      - pi = [pi_x, pi_y] is ith point
    //
    // OUTPUT:
    //   Finite element functions evaluated at points:
    //      function_at_points = [[f(p1), f(p2), ..., f(pM)],
    //                            [g(p1), g(p2), ..., g(pM)],
    //                            [h(p1), h(p2), ..., h(pM)]]
    //       - shape = (num_functions, num_pts)
    MatrixXd evaluate_functions_at_points( const Ref<const MatrixXd> functions_at_vertices, // shape=(num_functions, num_vertices)
                                           const Ref<const Matrix<double, K, Dynamic>> points ) // shape=(K, num_pts)
    {
        int num_functions = functions_at_vertices.rows();
        int num_pts = points.cols();

        MatrixXd function_at_points;
        function_at_points.resize(num_functions, num_pts);
        function_at_points.setZero();

//        Standard version (no multithreading)
//        ind_and_coords IC;
//        for ( int ii=0; ii<num_pts; ++ii )
//        {
//            get_simplex_ind_and_affine_coordinates_of_point( points.col(ii), IC );
//            if ( IC.simplex_ind >= 0 ) // point is in mesh
//            {
//                for ( int kk=0; kk<K+1; ++kk ) // for simplex vertex
//                {
//                    int vv = cells(kk, IC.simplex_ind);
//                    for ( int ll=0; ll<num_functions; ++ll ) // for each function
//                    {
//                        function_at_points(ll, ii) += IC.affine_coords(kk) * functions_at_vertices(ll, vv);
//                    }
//                }
//            }
//        }

        auto loop = [&](const int & start, const int & stop)
        {
            ind_and_coords<K> IC;
            for ( int ii=start; ii<stop; ++ii )
            {
                get_simplex_ind_and_affine_coordinates_of_point( points.col(ii), IC );
                if ( IC.simplex_ind >= 0 ) // point is in mesh
                {
                    for ( int kk=0; kk<K+1; ++kk ) // for simplex vertex
                    {
                        int vv = cells(kk, IC.simplex_ind);
                        for ( int ll=0; ll<num_functions; ++ll ) // for each function
                        {
                            function_at_points(ll, ii) += IC.affine_coords(kk) * functions_at_vertices(ll, vv);
                        }
                    }
                }
            }
        };

        pool.parallelize_loop(0, num_pts, loop);
//        cout << pool.get_thread_count() << endl;

        return function_at_points;
    }

    MatrixXd evaluate_functions_at_points_with_reflection( const Ref<const MatrixXd> functions_at_vertices, // shape=(num_functions, num_vertices)
                                                           const Ref<const Matrix<double, K, Dynamic>> points ) // shape=(K, num_pts)
    {
        int num_functions = functions_at_vertices.rows();
        int num_pts = points.cols();

        MatrixXd function_at_points;
        function_at_points.resize(num_functions, num_pts);
        function_at_points.setZero();

        auto loop = [&](const int & start, const int & stop)
        {
            ind_and_coords<K> IC;
            for ( int ii=start; ii<stop; ++ii )
            {
                KDVector point = points.col(ii);
                get_simplex_ind_and_affine_coordinates_of_point( point, IC );

                if ( IC.simplex_ind < 0 ) // if point is outside mesh
                {
                    point = 2.0 * closest_point( point ) - point; // reflect point across boundary
                    get_simplex_ind_and_affine_coordinates_of_point( point, IC );
                }

                if ( IC.simplex_ind >= 0 ) // if point is inside mesh
                {
                    for ( int kk=0; kk<K+1; ++kk ) // for simplex vertex
                    {
                        int vv = cells(kk, IC.simplex_ind);
                        for ( int ll=0; ll<num_functions; ++ll ) // for each function
                        {
                            function_at_points(ll, ii) += IC.affine_coords(kk) * functions_at_vertices(ll, vv);
                        }
                    }
                }
            }
        };

        pool.parallelize_loop(0, num_pts, loop);

        return function_at_points; // shape=(num_functions, num_points)
    }


    MatrixXd evaluate_functions_at_points_with_reflection_and_ellipsoid_truncation(
        const Ref<const MatrixXd>                   functions_at_vertices,                 // shape=(num_functions, num_vertices)
        const Ref<const Matrix<double, K, Dynamic>> points,                                // shape=(K, num_pts)
        const vector<KDVector> &                    ellipsoid_means,                       // size=num_functions
        const vector<Matrix<double, K, K>> &        ellipsoid_inverse_covariance_matrices, // size=num_functions
        double                                      ellipsoid_tau )
    {
        int num_functions = functions_at_vertices.rows();
        int num_pts = points.cols();

        double tau_squared = ellipsoid_tau * ellipsoid_tau;

        MatrixXd function_at_points;
        function_at_points.resize(num_functions, num_pts);
        function_at_points.setZero();

        auto loop = [&](const int & start, const int & stop)
        {
            ind_and_coords<K> IC;
            for ( int ii=start; ii<stop; ++ii )
            {
                KDVector point = points.col(ii);
                get_simplex_ind_and_affine_coordinates_of_point( point, IC );

                if ( IC.simplex_ind < 0 ) // if point is outside mesh, reflect it across the boundary
                {
                    point = 2.0 * closest_point( point ) - point; // reflection of point across boundary
                    get_simplex_ind_and_affine_coordinates_of_point( point, IC );
                }

                if ( IC.simplex_ind >= 0 ) // if point (whether original or reflected) is inside mesh
                {
                    vector<int> relevant_functions;
                    relevant_functions.reserve(num_functions);
                    for ( int ff=0; ff<num_functions; ++ff )
                    {
                        const KDVector & mu = ellipsoid_means[ff];
                        const Matrix<double, K, K> & M = ellipsoid_inverse_covariance_matrices[ff];
                        KDVector z = point - mu;
                        if ( z.transpose() * (M * z) < tau_squared )
                        {
                            relevant_functions.push_back(ff);
                        }
                    }

                    if ( ! relevant_functions.empty() )
                    {
                        for ( int kk=0; kk<K+1; ++kk ) // for simplex vertex
                        {
                            int vv = cells(kk, IC.simplex_ind);
                            for ( int ff : relevant_functions )
                            {
                                function_at_points(ff, ii) += IC.affine_coords(kk) * functions_at_vertices(ff, vv);
                            }
                        }
                    }

                }
            }
        };

        pool.parallelize_loop(0, num_pts, loop);

        return function_at_points; // shape=(num_functions, num_points)
    }

    void set_sleep_duration(int duration_in_microseconds)
    {
        pool.sleep_duration = duration_in_microseconds;
    }

    void reset_sleep_duration_to_default()
    {
        pool.sleep_duration = default_sleep_duration;
    }

    void set_thread_count(int num_threads)
    {
        pool.reset(num_threads);
    }

    void reset_thread_count_to_default()
    {
        pool.reset(default_number_of_threads);
    }

};

} // end namespace SMESH


