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
#include "geometric_sort.h"


namespace SMESH {

using namespace Eigen;
using namespace std;
using namespace KDT;
using namespace AABB;




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


class SimplexMesh
{
private:
    MatrixXi         faces;   // boundary face simplices of dimension dim-1. shape=(dim,num_faces)
    vector<VectorXi> subfaces; // sub-simplices of boundary faces of dimension 0 through dim-1 (includes boundary faces)

    vector<VectorXi> face2subface;

    AABBTree cell_aabbtree;
    AABBTree face_aabbtree;
    KDTree   face_kdtree;

    vector< Simplex > cell_simplices;
    vector< Simplex > subface_simplices;

    int dim;
    int num_vertices;
    int num_cells;
    int num_faces;
    int num_subfaces;

    int default_sleep_duration;
    int default_number_of_threads;

    std::pair<int, VectorXd> point_query_from_candidates(const VectorXi & candidate_inds, const VectorXd & point) const
    {
        int num_candidates = candidate_inds.size();

        int simplex_ind = -1;
        Eigen::VectorXd affine_coords(dim+1);
        for ( int jj=0; jj<num_candidates; ++jj ) // for each candidate simplex that the point might be in
        {
            int candidate_simplex_ind = candidate_inds(jj);
            const Simplex & S = cell_simplices[candidate_simplex_ind];
            affine_coords = S.A * point + S.b;
            if ( (affine_coords.array() >= 0.0).all() ) // point is in simplex
            {
                simplex_ind = candidate_simplex_ind;
                break;
            }
        }
        return std::make_pair(simplex_ind, affine_coords);
    }

    void eval_CG1_helper( MatrixXd &                  functions_at_points,
                          const vector<int> &         function_inds,
                          const vector<int> &         point_inds,
                          const VectorXi &            all_simplex_inds,
                          const MatrixXd &            all_affine_coords,
                          const Ref<const MatrixXd> & functions_at_vertices ) const
    {
        for ( int point_ind : point_inds )
        {
            int      simplex_ind   = all_simplex_inds(point_ind);
            VectorXd affine_coords = all_affine_coords.col(point_ind);

            for ( int kk=0; kk<dim+1; ++kk ) // for simplex vertex
            {
                int vv = cells(kk, simplex_ind);
                for ( int ll : function_inds ) // for each function
                {
                    functions_at_points(ll, point_ind) += affine_coords(kk) * functions_at_vertices(ll, vv);
                }
            }
        }
    }

public:
    MatrixXd    vertices; // shape=(dim,num_vertices)
    MatrixXi    cells;    // interior simplices of volumetric dimension. shape=(dim+1,num_cells)
    thread_pool pool;

    SimplexMesh( const Ref<const MatrixXd> input_vertices, // shape=(dim,num_vertices)
                 const Ref<const MatrixXi> input_cells )   // shape=(dim+1,num_cells)
    {
        // ------------------------    Input checking and copying    ------------------------
        dim = input_vertices.rows();
        num_vertices = input_vertices.cols();
        num_cells = input_cells.cols();

        if ( input_cells.rows() != dim+1 )
        {
            throw std::invalid_argument( "simplices have wrong dimension." );
        }

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
            MatrixXd simplex_vertices(dim, dim+1);
            for (int jj=0; jj<dim+1; ++jj )
            {
                simplex_vertices.col(jj) = vertices.col(cells(jj, ii));
            }
            std::pair<MatrixXd, VectorXd> STS = make_simplex_transform_operator( simplex_vertices );
            cell_simplices[ii] = Simplex { simplex_vertices, // V
                                           STS.first,        // A
                                           STS.second };     // b
        }

        // Generate cell AABB tree
        MatrixXd cell_box_mins (dim, num_cells);
        MatrixXd cell_box_maxes(dim, num_cells);
        for ( int ii=0; ii<num_cells; ++ii )
        {
            pair<VectorXd, VectorXd> BB = compute_pointcloud_bounding_box( cell_simplices[ii].V );
            cell_box_mins.col(ii) = BB.first;
            cell_box_maxes.col(ii) = BB.second;
        }
        cell_aabbtree.build_tree( cell_box_mins, cell_box_maxes );


        // ------------------------    FACES    ------------------------
        // For all faces (dim-1 dimensional simplex which has dim vertices), compute how many cells they are part of.
        map<vector<int>, int> face_counts; // face -> cell count
        for ( int cc=0; cc<num_cells; ++cc )
        {
            for ( int opposite_vertex_ind=0; opposite_vertex_ind<dim+1; ++opposite_vertex_ind )
            {
                vector<int> face;
                for ( int kk=0; kk<dim+1; ++kk )
                {
                    if ( kk != opposite_vertex_ind )
                    {
                        face.push_back(cells(kk, cc));
                    }
                }
                sort( face.begin(), face.end() ); // sort for comparison purposes

                if ( face_counts.find(face) == face_counts.end() ) // if this face isnt in the map yet
                {
                    face_counts[face] = 1;
                }
                else
                {
                    face_counts[face] += 1;
                }
            }
        }

        // Faces (faces on the boundary) are the faces that are part of only one cell
        vector<VectorXi> faces_vector;
        for ( auto it = face_counts.begin(); it != face_counts.end(); ++it )
        {
            vector<int> face = it->first;
            VectorXi F(dim);
            for ( int kk=0; kk<dim; ++kk)
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
        faces.resize(dim, num_faces);
        for ( int ii=0; ii<num_faces; ++ii )
        {
            faces.col(ii) = faces_vector[ii];
        }


        // Create kdtree of vertices that are on faces (i.e., on the boundary)
        set<int> face_vertex_inds;
        for ( int bb=0; bb<num_faces; ++bb )
        {
            for ( int kk=0; kk<dim; ++kk )
            {
                face_vertex_inds.insert(faces(kk, bb));
            }
        }

        int num_face_vertices = face_vertex_inds.size();
        MatrixXd face_vertices(dim,num_face_vertices);
        int vv=0;
        for ( auto it  = face_vertex_inds.begin();
                   it != face_vertex_inds.end();
                 ++it )
        {
            face_vertices.col(vv) = vertices.col( *it );
            vv += 1;
        }
        face_kdtree.build_tree( face_vertices );


        // Create face AABB tree
        MatrixXd face_box_mins(dim, num_faces);
        MatrixXd face_box_maxes(dim, num_faces);
        for ( int bb=0; bb<num_faces; ++bb )
        {
            MatrixXd face_vertices(dim,dim);
            for (int jj=0; jj<dim; ++jj )
            {
                face_vertices.col(jj) = vertices.col(faces(jj, bb));
            }
            pair<VectorXd, VectorXd> BB = compute_pointcloud_bounding_box( face_vertices );
            face_box_mins.col(bb) = BB.first;
            face_box_maxes.col(bb) = BB.second;
        }
        face_aabbtree.build_tree( face_box_mins, face_box_maxes );


        // ------------------------    SUBFACES    ------------------------
        // Construct all boundary entities (faces-of-faces, etc)
        vector<vector<int>> pset = powerset(dim); // powerset(3) = [[], [0], [1], [0, 1], [2], [0, 2], [1, 2], [0, 1, 2]]
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
            MatrixXd subface_vertices(dim, num_vertices_in_subface);
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

    inline bool point_is_in_mesh( const VectorXd & query ) const
    {
        return (point_query( query ).first >= 0);
    }

    Matrix<bool, Dynamic, 1> point_is_in_mesh_vectorized( const Ref<const MatrixXd> query_points ) const
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

    VectorXd closest_point( const VectorXd & query ) const
    {
        VectorXd point = vertices.col(0);
        if ( point_is_in_mesh( query ) )
        {
            point = query;
        }
        else
        {
            // 1. Find a set of candidate boundary faces, one of which contains the closest point
            pair<VectorXi, VectorXd> kd_result = face_kdtree.query( query, 1 );
            double dist_estimate = (1.0 + 1e-14) * sqrt(kd_result.second(0));
            VectorXi face_inds = face_aabbtree.ball_collisions( query, dist_estimate );

            // 2. Determine unique set of boundary entities to visit
            vector<int> entities;
            entities.reserve(power_of_two(dim));
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
            double dsq_best = (point - query).squaredNorm();
            for ( int ee=0; ee<entities.size(); ++ee )
            {
                const Simplex & E = subface_simplices[entities[ee]];
                VectorXd projected_affine_coords = E.A * query + E.b;
                if ( (projected_affine_coords.array() >= 0.0).all() ) // projection is in subface simplex
                {
                    VectorXd projected_query = E.V * projected_affine_coords;
                    double dsq = (projected_query - query).squaredNorm();
                    if ( dsq < dsq_best )
                    {
                        point = projected_query;
                        dsq_best = dsq;
                    }
                }
            }
        }
        return point;
    }

    MatrixXd closest_point_vectorized( const Ref<const MatrixXd> query_points )
    {
        int num_queries = query_points.cols();
        MatrixXd closest_points;
        closest_points.resize(dim, num_queries);

        std::vector<int> shuffle_inds(num_queries); // randomize ordering to make work even among threads
        std::iota(shuffle_inds.begin(), shuffle_inds.end(), 0);
        std::random_shuffle(shuffle_inds.begin(), shuffle_inds.end());

        auto loop = [&](const int &a, const int &b)
        {
            for ( int ii=a; ii<b; ++ii )
            {
                closest_points.col(shuffle_inds[ii]) = closest_point( query_points.col(shuffle_inds[ii]) );
            }
        };

        pool.parallelize_loop(0, num_queries, loop);
        return closest_points;
    }

    std::pair<int,VectorXd> point_query( const Eigen::VectorXd & point ) const
    {
        VectorXi candidate_inds =  cell_aabbtree.point_collisions( point );
        return point_query_from_candidates(candidate_inds, point);
    }

    std::pair<VectorXi,MatrixXd> point_query_vectorized( const Eigen::MatrixXd & points )
    {
        int num_pts = points.cols();

        std::vector<Eigen::VectorXi> all_candidate_inds = cell_aabbtree.point_collisions_vectorized( points );

        VectorXi all_simplex_inds(num_pts);
        MatrixXd all_affine_coords(dim+1, num_pts);

        std::vector<int> shuffle_inds(num_pts); // randomize ordering to make work even among threads
        std::iota(shuffle_inds.begin(), shuffle_inds.end(), 0);
        std::random_shuffle(shuffle_inds.begin(), shuffle_inds.end());

        auto loop = [&](const int &a, const int &b)
        {
            for ( int ii=a; ii<b; ++ii )
            {
                int ind = shuffle_inds[ii];
                std::pair<int,VectorXd> IC = point_query_from_candidates(all_candidate_inds[ind],
                                                                         points        .col(ind));
                all_simplex_inds     [ind] = IC.first;
                all_affine_coords.col(ind) = IC.second;
            }
        };

        pool.parallelize_loop(0, num_pts, loop);
        return make_pair(all_simplex_inds, all_affine_coords);
    }


    // ------------    SimplexMesh::evaluate_CG1_functions_at_points()    --------------
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
    //      - shape = (dim, num_pts)
    //      - dim = spatial dimension
    //      - pi = [pi_x, pi_y] is ith point
    //
    // OUTPUT:
    //   Finite element functions evaluated at points:
    //      function_at_points = [[f(p1), f(p2), ..., f(pM)],
    //                            [g(p1), g(p2), ..., g(pM)],
    //                            [h(p1), h(p2), ..., h(pM)]]
    //       - shape = (num_functions, num_pts)
    MatrixXd evaluate_CG1_functions_at_points( const Ref<const MatrixXd> functions_at_vertices, // shape=(num_functions, num_vertices)
                                               const Ref<const MatrixXd> points ) // shape=(dim, num_pts)
    {
        int num_functions = functions_at_vertices.rows();
        int num_pts = points.cols();

        MatrixXd functions_at_points(num_functions, num_pts);
        functions_at_points.setZero();

//        std::pair<VectorXi,MatrixXd> ICS = point_query_vectorized( points );
        VectorXi all_simplex_inds(num_pts);
        MatrixXd all_affine_coords(dim+1, num_pts);

        std::vector<int> function_inds(num_functions);
        std::iota(function_inds.begin(), function_inds.end(), 0);

        auto loop = [&](const int & start, const int & stop)
        {
            vector<int> point_inds;
            point_inds.reserve(stop-start);

            for ( int ii=start; ii<stop; ++ii )
            {
                std::pair<int,VectorXd> IC = point_query( points.col(ii) );
                all_simplex_inds[ii] = IC.first;
                all_affine_coords.col(ii) = IC.second;
                if ( IC.first >= 0 ) // if point is inside mesh
                {
                    point_inds.push_back(ii);
                }
            }

            eval_CG1_helper( functions_at_points,
                             function_inds,
                             point_inds,
                             all_simplex_inds,
                             all_affine_coords,
                             functions_at_vertices );
        };

        pool.parallelize_loop(0, num_pts, loop);
        return functions_at_points;
    }

//    MatrixXd evaluate_functions_at_points_with_reflection( const Ref<const MatrixXd> functions_at_vertices, // shape=(num_functions, num_vertices)
//                                                           const Ref<const MatrixXd> points ) // shape=(dim, num_pts)
//    {
//        int num_functions = functions_at_vertices.rows();
//        int num_pts = points.cols();
//
//        MatrixXd functions_at_points(num_functions, num_pts);
//        functions_at_points.setZero();
//
//        auto loop = [&](const int & start, const int & stop)
//        {
//            vector<int>      all_point_inds;    all_point_inds   .reserve(stop-start);
//            vector<int>      all_simplex_inds;  all_simplex_inds .reserve(stop-start);
//            vector<VectorXd> all_affine_coords; all_affine_coords.reserve(stop-start);
//            for ( int ii=start; ii<stop; ++ii )
//            {
//                VectorXd point = points.col(ii);
//                std::pair<int,VectorXd> IC = point_query( point );
//                if ( IC.first < 0 ) // if point is outside mesh
//                {
//                    point = 2.0 * closest_point( point ) - point; // reflect point across boundary
//                    IC = point_query( point );
//                }
//                if ( IC.first >= 0 ) // if point is inside mesh
//                {
//                    all_point_inds.   push_back(ii);
//                    all_simplex_inds. push_back(IC.first);
//                    all_affine_coords.push_back(IC.second);
//                }
//            }
//
//            std::vector<int> function_inds(num_functions);
//            std::iota(function_inds.begin(), function_inds.end(), 0);
//
//            eval_CG1_helper( functions_at_points,
//                             function_inds,
//                             all_point_inds,
//                             all_simplex_inds,
//                             all_affine_coords,
//                             functions_at_vertices );
//        };
//
//        pool.parallelize_loop(0, num_pts, loop);
//        return functions_at_points; // shape=(num_functions, num_points)
//    }


    MatrixXd evaluate_functions_at_points_with_reflection_and_ellipsoid_truncation(
        const Ref<const MatrixXd>                   functions_at_vertices,                 // shape=(num_functions, num_vertices)
        const Ref<const MatrixXd>                   points,                                // shape=(dim, num_pts)
        const vector<VectorXd> &                    ellipsoid_means,                       // size=num_functions
        const vector<MatrixXd> &        ellipsoid_inverse_covariance_matrices, // size=num_functions, each shape=(dim,dim)
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
            for ( int ii=start; ii<stop; ++ii )
            {
                VectorXd point = points.col(ii);
                std::pair<int,VectorXd> IC = point_query( point );

                if ( IC.first < 0 ) // if point is outside mesh, reflect it across the boundary
                {
                    point = 2.0 * closest_point( point ) - point; // reflection of point across boundary
                    IC = point_query( point );
                }

                if ( IC.first >= 0 ) // if point (whether original or reflected) is inside mesh
                {
                    vector<int> relevant_functions;
                    relevant_functions.reserve(num_functions);
                    for ( int ff=0; ff<num_functions; ++ff )
                    {
                        const VectorXd & mu = ellipsoid_means[ff];
                        const MatrixXd & M  = ellipsoid_inverse_covariance_matrices[ff];
                        VectorXd z = point - mu;
                        if ( z.transpose() * (M * z) < tau_squared )
                        {
                            relevant_functions.push_back(ff);
                        }
                    }

                    if ( ! relevant_functions.empty() )
                    {
                        for ( int kk=0; kk<dim+1; ++kk ) // for simplex vertex
                        {
                            int vv = cells(kk, IC.first);
                            for ( int ff : relevant_functions )
                            {
                                function_at_points(ff, ii) += IC.second(kk) * functions_at_vertices(ff, vv);
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


