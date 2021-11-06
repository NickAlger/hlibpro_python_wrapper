#pragma once

#include <iostream>
#include <list>
#include <queue>

#include <math.h>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

// Nearest neighbor in subtree to query point
struct SubtreeResult
{
    int index; // index of nearest neighbor
    double distance_squared; // distance squared to query point

    const bool operator < ( const SubtreeResult& other ) const
    {
        return ( distance_squared < other.distance_squared );
    }
};

// Node in KD tree
template <int K>
struct KDNode { Matrix<double,K,1> point;
              int                left;       // index of left child
              int                right; };   // index of right child

template <int K>
struct PointWithIndex { Matrix<double,K,1> point;
                        int index; };

template <int K>
class KDTree {
private:
    vector< KDNode<K> > nodes; // All nodes in the tree
    Matrix<int, Dynamic, 1> perm_i2e; // permutation from internal ordering to external ordering

    // creates subtree and returns the index for root of subtree
    int make_subtree( int start, int stop, int depth,
                      vector< PointWithIndex<K> > & points,
                      int & counter )
    {
        int num_pts_local = stop - start;
        int current_node_ind = -1; // -1 indicates node does not exist
        if (num_pts_local >= 1)
        {
            current_node_ind = counter;
            counter = counter + 1;

            int axis = depth % K;
            sort( points.begin() + start, points.begin() + stop,
                  [axis](PointWithIndex<K> u, PointWithIndex<K> v) {return u.point(axis) > v.point(axis);} );

            int mid = start + (num_pts_local / 2);

            int left_start = start;
            int left_stop = mid;

            int right_start = mid + 1;
            int right_stop = stop;

            int left = make_subtree(left_start, left_stop, depth + 1, points, counter);
            int right = make_subtree(right_start, right_stop, depth + 1, points, counter);

            nodes[current_node_ind] = KDNode<K> { points[mid].point, left, right };
            perm_i2e[current_node_ind] = points[mid].index;
        }
        return current_node_ind;
    }

    // finds nearest neighbor of query in subtree
    SubtreeResult nn_subtree( const Matrix<double,K,1> & query,
                              int                        root_index,
                              int                        depth) {
        KDNode<K> root = nodes[root_index];

        Matrix<double,K,1> delta = query - root.point;

        int best_index = root_index;
        double best_distance_squared = delta.squaredNorm();

        int axis = depth % K;
        double displacement_to_splitting_plane = delta(axis);

        int A;
        int B;
        if (displacement_to_splitting_plane >= 0) {
            A = root.left;
            B = root.right;
        } else {
            A = root.right;
            B = root.left; }

        if (A >= 0) {
            SubtreeResult nn_A = nn_subtree( query, A, depth + 1);
            if (nn_A.distance_squared < best_distance_squared) {
                best_index = nn_A.index;
                best_distance_squared = nn_A.distance_squared; } }

        if (B >= 0) {
            bool nearest_neighbor_might_be_in_B_subtree =
                displacement_to_splitting_plane*displacement_to_splitting_plane < best_distance_squared;
            if (nearest_neighbor_might_be_in_B_subtree) {
                SubtreeResult nn_B = nn_subtree( query, B, depth + 1);
                if (nn_B.distance_squared < best_distance_squared) {
                    best_index = nn_B.index;
                    best_distance_squared = nn_B.distance_squared; } } }

        return SubtreeResult { best_index, best_distance_squared }; }


    // finds num_neighbors nearest neighbors of query in subtree
    void nn_subtree_many( const Matrix<double,K,1> &                                query,
                          priority_queue<SubtreeResult, vector<SubtreeResult>> &    nn,
                          int                                                       cur_index,
                          int                                                       depth,
                          int                                                       num_neighbors )
    {
        KDNode<K> cur = nodes[cur_index];

        Matrix<double,K,1> delta = query - cur.point;
        double dsq_cur = delta.squaredNorm();
        SubtreeResult cur_result = {cur_index, dsq_cur};

        if ( nn.size() < num_neighbors )
        {
            nn.push( cur_result );
        }
        else if ( dsq_cur < nn.top().distance_squared )
        {
            nn.pop();
            nn.push( cur_result );
        }

        int axis = depth % K;
        double displacement_to_splitting_plane = delta(axis);

        int A;
        int B;
        if (displacement_to_splitting_plane >= 0)
        {
            A = cur.left;
            B = cur.right;
        }
        else
        {
            A = cur.right;
            B = cur.left;
        }

        if (A >= 0)
        {
            nn_subtree_many( query, nn, A, depth+1, num_neighbors );
        }

        if (B >= 0)
        {
            if ( displacement_to_splitting_plane*displacement_to_splitting_plane
                 < nn.top().distance_squared )
            {
                nn_subtree_many( query, nn, B, depth+1, num_neighbors );
            }
        }
    }

public:
    KDTree( ) {}

    KDTree( const Ref<const Matrix<double, K, Dynamic>> points_array )
    {
        int num_pts = points_array.cols();

        // Copy eigen matrix input into std::vector of tuples which will be re-ordered
        vector< PointWithIndex<K> > points(num_pts); // (coords, original_index)
        for ( int ii=0; ii<num_pts; ++ii)
        {
            points[ii].point = points_array.col(ii);
            points[ii].index = ii;
        }

        nodes.reserve(num_pts);
        perm_i2e.resize(num_pts, 1);
        int counter = 0;
        int zero = make_subtree(0, num_pts, 0, points, counter);
    }

    pair<int, double> nearest_neighbor( const Matrix<double,K,1> & point )
    {
        SubtreeResult nn_result = nn_subtree( point, 0, 0 );
        return make_pair(perm_i2e[nn_result.index], nn_result.distance_squared);
    }

    pair<VectorXi, VectorXd> nearest_neighbors( const Matrix<double,K,1> & point, int num_neighbors )
    {
        vector<SubtreeResult> nn_container;
        nn_container.reserve(2*num_neighbors);
        priority_queue<SubtreeResult, vector<SubtreeResult>> nn(less<SubtreeResult>(), move(nn_container));

        nn_subtree_many( point, nn, 0, 0, num_neighbors );

        VectorXi nn_inds(num_neighbors);
        VectorXd nn_dsq(num_neighbors);
        for ( int ii=0; ii<num_neighbors; ++ii )
        {
            int jj = num_neighbors - ii - 1;
            SubtreeResult n_ii = nn.top();
            nn.pop();
            nn_inds(jj) = perm_i2e[n_ii.index];
            nn_dsq(jj) = n_ii.distance_squared;
        }
        return make_pair(nn_inds, nn_dsq);
    }

    pair< MatrixXi, MatrixXd >
        nearest_neighbors_vectorized( const Ref<const Matrix<double, K, Dynamic>> query_array,
                                      int num_neighbors )
    {
        int num_querys = query_array.cols();

        MatrixXi closest_point_inds(num_neighbors, num_querys);
        MatrixXd squared_distances(num_neighbors, num_querys);

        for ( int ii=0; ii<num_querys; ++ii )
        {
            pair<VectorXi, VectorXd> result = nearest_neighbors( query_array.col(ii), num_neighbors );
            closest_point_inds.col(ii) = result.first;
            squared_distances.col(ii) = result.second;
        }
        return make_pair(closest_point_inds, squared_distances);
    }

    pair< VectorXi, VectorXd >
        nearest_neighbor_vectorized( const Ref<const Matrix<double, K, Dynamic>> query_array )
    {
        int num_querys = query_array.cols();

        VectorXi closest_points_inds(num_querys);

        VectorXd squared_distances(num_querys);

        for ( int ii=0; ii<num_querys; ++ii )
        {
            Matrix<double,K,1> query = query_array.col(ii);
            SubtreeResult nn_result = nn_subtree( query, 0, 0 );
            closest_points_inds(ii) = perm_i2e[nn_result.index];
            squared_distances(ii) = nn_result.distance_squared;
        }

        return make_pair(closest_points_inds, squared_distances);
    }
};

