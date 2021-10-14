#include <iostream>
#include <list>

#include <math.h>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;


template <int K>
class AABBTree {
private:
    typedef Array<double, K, 1> KDVector;

    struct Box { KDVector min;
                 KDVector max;
                 int      index;}

    // Node in KD tree
    struct Node { Box box;
                  int left;       // index of left child
                  int right; };   // index of right child

    vector< Node > nodes; // All nodes in the tree

    // creates subtree and returns the index for root of subtree
    int make_subtree( int start, int stop,
                      vector<Box> & leaf_boxes,
                      int & counter ) {
        int num_boxes_local = stop - start;
        int current_node_ind = -1; // -1 indicates node does not exist
        if (num_boxes_local >= 1) {
            current_node_ind = counter;
            counter = counter + 1;

            // Find big box containing all boxes in this group
            KDVector big_box_min;
            for ( int kk=0; kk<K; ++kk )
            {
                double min_k = leaf_boxes[start].min(kk);
                for ( int bb=start+1; bb<stop; ++bb)
                {
                    double candidate_min_k = leaf_boxes[bb].min(kk);
                    if (candidate_min_k < min_k)
                    {
                        min_k = candidate_min_k;
                    }
                }
                big_box_min(kk) = min_k;
            }

            KDVector big_box_max;
            for ( int kk=0; kk<K; ++kk )
            {
                double max_k = leaf_boxes[start].max(kk);
                for ( int bb=start+1; bb<stop; ++bb)
                {
                    double candidate_max_k = leaf_boxes[bb].max(kk);
                    if (candidate_max_k > max_k)
                    {
                        max_k = candidate_max_k;
                    }
                }
                big_box_max(kk) = max_k;
            }

            Box big_box {big_box_min, big_box_max, -1};

            // Find biggest axis of box
            int axis = 0;
            double biggest_axis_size = big_box_max(0) - big_box_min(0);
            for ( int kk=1; kk<K; ++kk)
            {
                double kth_axis_size = big_box_max(kk) - big_box_min(kk);
                if (kth_axis_size > biggest_axis_size)
                {
                    axis = k;
                    biggest_axis_size = kth_axis_size;
                }
            }

            // Sort boxes by centerpoint along biggest axis
            sort( leaf_boxes.begin() + start,
                  leaf_boxes.begin() + stop,
                  [axis](Box A, Box B) {return 0.5*(A.max(axis)+A.min(axis))
                                             > 0.5*(B.max(axis)+B.min(axis));} ); // 0.5*(...) for clarity (unnecessary)

            // Find index of first box on "right" side of biggest axis
            double centerpoint = 0.5*(big_box.max(axis) + big_box.min(axis));
            int mid = 0;
            for ( int bb=0; bb<num_boxes_local; ++bb )
            {
                double box_centerpoint = 0.5*(leaf_boxes[bb].max(axis) + leaf_boxes[bb].min(axis));
                if ( box_centerpoint < centerpoint )
                {
                    mid = bb+1;
                }
            }

            int left  = make_subtree(start,    mid, leaf_boxes, counter);
            int right = make_subtree(mid + 1, stop, leaf_boxes, counter);

            Node big_box_node { big_box, left, right };
            nodes.push_back(big_box_node);
            }
        return current_node_ind;
        }

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
    KDTree( Array<double, Dynamic, K> & points_array ) {
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

