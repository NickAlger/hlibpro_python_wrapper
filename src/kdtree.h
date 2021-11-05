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

//bool compare_subtree_results(SubtreeResult r1, SubtreeResult r2){return r1.distance_squared < r2.distance_squared;}
bool compare_subtree_result_to_distance_squared(SubtreeResult r, double dsq){return r.distance_squared < dsq;}

template <int K>
class KDTree {
private:
    typedef Matrix<double, K, 1> KDVector;

    // Node in KD tree
    struct Node { KDVector point;
                  int      left;       // index of left child
                  int      right; };   // index of right child

    vector< Node > nodes; // All nodes in the tree

    // creates subtree and returns the index for root of subtree
    int make_subtree( int start, int stop, int depth,
                      vector< KDVector > & points,
                      int & counter ) {
        int num_pts_local = stop - start;
        int current_node_ind = -1; // -1 indicates node does not exist
        if (num_pts_local >= 1) {
            current_node_ind = counter;
            counter = counter + 1;

            int axis = depth % K;
            sort( points.begin() + start, points.begin() + stop,
                  [axis](KDVector u, KDVector v) {return u(axis) > v(axis);} );

            int mid = start + (num_pts_local / 2);

            int left_start = start;
            int left_stop = mid;

            int right_start = mid + 1;
            int right_stop = stop;

            int left = make_subtree(left_start, left_stop, depth + 1, points, counter);
            int right = make_subtree(right_start, right_stop, depth + 1, points, counter);

            nodes[current_node_ind] = Node { points[mid], left, right }; }
        return current_node_ind; }

    // finds nearest neighbor of query in subtree
    SubtreeResult nn_subtree( const KDVector & query,
                              int              root_index,
                              int              depth) {
        Node root = nodes[root_index];

        KDVector delta = query - root.point;

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
    void nn_subtree_many( const KDVector &                   query,
                          priority_queue<SubtreeResult> &    nn,
                          int                                cur_index,
                          int                                depth,
                          int                                num_neighbors )
    {
        Node cur = nodes[cur_index];

        KDVector delta = query - cur.point;
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
        vector< KDVector > points(num_pts);
        for ( int ii=0; ii<num_pts; ++ii)
        {
            points[ii] = points_array.col(ii);
        }

        nodes.reserve(num_pts);
        int counter = 0;
        int zero = make_subtree(0, num_pts, 0, points, counter);
    }

    pair<KDVector, double> nearest_neighbor( const KDVector & point )
    {
        SubtreeResult nn_result = nn_subtree( point, 0, 0 );
        return make_pair(nodes[nn_result.index].point,
                         nn_result.distance_squared);
    }

    pair<MatrixXd, VectorXd> nearest_neighbors( const KDVector & point, int num_neighbors )
    {
        priority_queue<SubtreeResult> nn;
        nn_subtree_many( point, nn, 0, 0, num_neighbors);

        MatrixXd nn_vectors(K, num_neighbors);
        VectorXd nn_dsq(num_neighbors);
        for ( int ii=0; ii<num_neighbors; ++ii )
        {
            int jj = num_neighbors - ii - 1;
            SubtreeResult n_ii = nn.top();
            nn.pop();
            nn_vectors.col(jj) = nodes[n_ii.index].point;
            nn_dsq(jj) = n_ii.distance_squared;
        }
        return make_pair(nn_vectors, nn_dsq);
    }

    pair< vector<Matrix<double, K, Dynamic>>, vector<VectorXd> >
        nearest_neighbors_vectorized( const Ref<const Matrix<double, K, Dynamic>> query_array,
                                      int num_neighbors )
    {
        int num_querys = query_array.cols();

        vector<Matrix<double, K, Dynamic>> closest_points(num_querys);
        vector<VectorXd> squared_distances(num_querys);

        for ( int ii=0; ii<num_querys; ++ii )
        {
            pair<MatrixXd, VectorXd> result = nearest_neighbors( query_array.col(ii), num_neighbors );
            closest_points[ii] = result.first;
            squared_distances[ii] = result.second;
        }
        return make_pair(closest_points, squared_distances);
    }

    pair< Matrix<double, K, Dynamic>, VectorXd >
        nearest_neighbor_vectorized( const Ref<const Matrix<double, K, Dynamic>> query_array )
    {
        int num_querys = query_array.cols();

        Matrix<double, K, Dynamic> closest_points_array(K, num_querys);

        VectorXd squared_distances(num_querys);

        for ( int ii=0; ii<num_querys; ++ii )
        {
            KDVector query = query_array.col(ii);
            SubtreeResult nn_result = nn_subtree( query, 0, 0 );
            closest_points_array.col(ii) = nodes[nn_result.index].point;
            squared_distances(ii) = nn_result.distance_squared;
        }

        return make_pair(closest_points_array, squared_distances);
    }
};

