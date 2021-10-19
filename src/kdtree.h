#pragma once

#include <iostream>
#include <list>

#include <math.h>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;


template <int K>
class KDTree {
private:
    typedef Matrix<double, K, 1> KDVector;

    // Node in KD tree
    struct Node { KDVector point;
                  int      left;       // index of left child
                  int      right; };   // index of right child

    // Nearest neighbor in subtree to query point
    struct SubtreeResult { int index; // index of nearest neighbor
                           double distance_squared; }; // distance squared to nearest neighbor

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

