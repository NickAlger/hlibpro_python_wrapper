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
                 int      index;}; // 0,...,N for leaf nodes, and -1 for internal node

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
        if ( num_boxes_local == 1)
        {
            current_node_ind = counter;
            counter = counter + 1;
            nodes[current_node_ind] = Node { leaf_boxes[start], -1, -1 };
        }
        else if (num_boxes_local > 1)
        {
            current_node_ind = counter;
            counter = counter + 1;

            // compute limits of big box containing all boxes in this group
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
                    axis = kk;
                    biggest_axis_size = kth_axis_size;
                }
            }

            // Sort boxes by centerpoint along biggest axis
            sort( leaf_boxes.begin() + start,
                  leaf_boxes.begin() + stop,
                  [axis](Box A, Box B) {return 0.5*(A.max(axis)+A.min(axis))
                                             > 0.5*(B.max(axis)+B.min(axis));} ); // 0.5*(...) for clarity (unnecessary)

            // Find index of first leaf box with centerpoint in the "right" half of big box
            double big_box_centerpoint = 0.5*(big_box.max(axis) + big_box.min(axis));
            int mid = stop;
            for ( int bb=start; bb<stop; ++bb )
            {
                Box L = leaf_boxes[bb];
                double L_centerpoint = 0.5*(L.max(axis) + L.min(axis));
                if ( big_box_centerpoint < L_centerpoint )
                {
                    mid = bb;
                    break;
                }
            }

            // If all boxes happen to be on one side, split them evenly
            // (theoretically possible, e.g., if all boxes have the same centerpoint)
            if ( (mid == start) || (mid == stop) )
            {
                mid = start + (num_boxes_local / 2);
            }

            int left  = make_subtree(start,  mid, leaf_boxes, counter);
            int right = make_subtree(mid,   stop, leaf_boxes, counter);

            nodes[current_node_ind] = Node { big_box, left, right };
            }
        return current_node_ind;
        }


    // finds nearest neighbor of query in subtree
    int first_point_intersection_subtree( const KDVector & query,
                                          int              current )
    {
        Node current_node = nodes[current];
        Box B = current_node.box;

        bool query_is_in_current_box = true;
        for ( int kk=0; kk<K; ++kk)
        {
            if ( (query(kk) < B.min(kk)) || (B.max(kk) < query(kk)) )
            {
                query_is_in_current_box = false;
                break;
            }
        }

        int first_intersection = -1;
        if ( query_is_in_current_box )
        {
            if ( B.index >= 0 ) // if box is leaf
            {
                first_intersection = B.index;
            } else { // box is internal node
                first_intersection = first_point_intersection_subtree( query, current_node.left );
                if (first_intersection < 0 ) // point didn't intersect with left subtree
                {
                    first_intersection = first_point_intersection_subtree( query, current_node.right );
                }
            }
        }

        return first_intersection;
    }

public:
    AABBTree( Array<double, Dynamic, K> & box_mins,
              Array<double, Dynamic, K> & box_maxes )
    {
        int num_leaf_boxes = box_mins.rows();

        // Copy eigen matrix input into std::vector of Boxes which will be re-ordered
        vector< Box > leaf_boxes(num_leaf_boxes);
        for ( int ii=0; ii<num_leaf_boxes; ++ii)
        {
            leaf_boxes[ii] = Box { box_mins.row(ii), box_maxes.row(ii), ii };
        }

        int num_boxes = 2*num_leaf_boxes - 1; // full binary tree with n leafs has 2n-1 nodes
        nodes.reserve(num_boxes);
        int counter = 0;
        int zero = make_subtree(0, num_leaf_boxes, leaf_boxes, counter);
    }

    int first_point_intersection( KDVector query )
    {
        return first_point_intersection_subtree( query, 0 );
    }

    VectorXi first_point_intersection_vectorized( Array<double, Dynamic, K> & query_array )
    {
        int num_querys = query_array.rows();
        VectorXi first_intersection_inds(num_querys);
        for (int ii=0; ii<num_querys; ++ii)
        {
            KDVector query = query_array.row(ii);
            first_intersection_inds(ii) = first_point_intersection_subtree( query, 0 );
        }
        return first_intersection_inds;
    }

};

