#pragma once

#include <iostream>
#include <list>

#include <math.h>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

// Nearest neighbor in subtree to query point
struct SubtreeResult { int index; // index of nearest neighbor
                       double distance_squared; }; // distance squared to nearest neighbor

bool compare_subtree_results(SubtreeResult r1, SubtreeResult r2){return r1.distance_squared < r2.distance_squared;}
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
    vector<SubtreeResult> nn_subtree_many( const KDVector & query,
                                           int              root_index,
                                           int              depth,
                                           int              num_neighbors )
    {
        Node root = nodes[root_index];

        KDVector delta = query - root.point;

        vector<SubtreeResult> nn;

        if ( (root.left < 0) && (root.right < 0) ) // root is a leaf
        {
            nn.push_back(SubtreeResult {root_index, delta.squaredNorm()});
        }
        else // this node is not a leaf
        {
            if ( (root.left >= 0) && (root.right >= 0) ) // root has two child nodes
            {
                nn.reserve(2*num_neighbors + 1);
            }
            else // root has one child node
            {
                nn.reserve(num_neighbors + 1);
            }
            nn.push_back(SubtreeResult {root_index, delta.squaredNorm()});

            int axis = depth % K;
            double displacement_to_splitting_plane = delta(axis);

            int A;
            int B;
            if (displacement_to_splitting_plane >= 0)
            {
                A = root.left;
                B = root.right;
            }
            else
            {
                A = root.right;
                B = root.left;
            }

            if (A >= 0)
            {
                vector<SubtreeResult> nn_A = nn_subtree_many( query, A, depth + 1, num_neighbors );
                nn.insert( nn.end(), nn_A.begin(), nn_A.end() );
                inplace_merge(nn.begin(), nn.begin()+1, nn.end(), compare_subtree_results);
            }

            if (B >= 0)
            {
                double displacement_squared = displacement_to_splitting_plane*displacement_to_splitting_plane;
                int num_good = lower_bound(nn.begin(), nn.end(),
                                           displacement_squared,
                                           compare_subtree_result_to_distance_squared) - nn.begin();
                int num_neighbors_B = num_neighbors - num_good;
                if (num_neighbors_B > 0)
                {
                    vector<SubtreeResult> nn_B = nn_subtree_many( query, B, depth + 1, num_neighbors_B );
                    int size0 = nn.size();
                    nn.insert( nn.end(), nn_B.begin(), nn_B.end() );
                    inplace_merge(nn.begin(), nn.begin()+size0, nn.end(), compare_subtree_results);
                }
            }
        }

        if (nn.size() > num_neighbors)
        {
            nn.resize(num_neighbors);
        }
        return nn;
    }

public:
    KDTree( ) {}

    KDTree( const Ref<const Array<double, Dynamic, K>> points_array ) {
        int num_pts = points_array.rows();

        // Copy eigen matrix input into std::vector of tuples which will be re-ordered
        vector< KDVector > points(num_pts);
        for ( int ii=0; ii<num_pts; ++ii) {
            points[ii] = points_array.row(ii); }

        nodes.reserve(num_pts);
        int counter = 0;
        int zero = make_subtree(0, num_pts, 0, points, counter); }

    pair<KDVector, double> nearest_neighbor( KDVector point ) {
        SubtreeResult nn_result = nn_subtree( point, 0, 0 );
        return make_pair(nodes[nn_result.index].point,
                         nn_result.distance_squared); }

    pair<MatrixXd, VectorXd> nearest_neighbors( KDVector point, int num_neighbors )
    {
        vector<SubtreeResult> nn = nn_subtree_many( point, 0, 0, num_neighbors );
        MatrixXd nn_vectors(K, num_neighbors);
        VectorXd nn_dsq(num_neighbors);
        for ( int ii=0; ii<num_neighbors; ++ii )
        {
            nn_vectors.col(ii) = nodes[nn[ii].index].point;
            nn_dsq(ii) = nn[ii].distance_squared;
        }
        return make_pair(nn_vectors, nn_dsq);
    }

    pair< Array<double, Dynamic, K>, VectorXd >
        nearest_neighbor_vectorized( Array<double, Dynamic, K> & query_array ) {
        int num_querys = query_array.rows();

        Array<double, Dynamic, K> closest_points_array;
        closest_points_array.resize(num_querys, K);

        VectorXd squared_distances(num_querys);

        for ( int ii=0; ii<num_querys; ++ii ) {
            KDVector query = query_array.row(ii);
            SubtreeResult nn_result = nn_subtree( query, 0, 0 );
            closest_points_array.row(ii) = nodes[nn_result.index].point;
            squared_distances(ii) = nn_result.distance_squared; }

        return make_pair(closest_points_array, squared_distances); } };

