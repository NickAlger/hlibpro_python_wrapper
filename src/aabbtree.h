#pragma once

#include <iostream>
#include <list>

#include <math.h>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;


template <int K>
class AABBTree {
private:
    typedef Matrix<double, K, 1> KDVector;

    struct Box { KDVector min;
                 KDVector max;
                 int      index;}; // 0,...,N for leaf nodes, and -1 for internal node

    // Node in KD tree
    struct Node { Box box;
                  int left;       // index of left child
                  int right; };   // index of right child

    vector< Node > nodes; // All nodes in the tree
    vector< int > nodes_under_consideration;

    double box_center( const Box & B, int axis )
    {
        return 0.5*(B.max(axis)+B.min(axis));
    }

    int biggest_axis_of_box( const Box & B )
    {
        int axis = 0;
        double biggest_axis_size = B.max(0) - B.min(0);
        for ( int kk=1; kk<K; ++kk)
        {
            double kth_axis_size = B.max(kk) - B.min(kk);
            if (kth_axis_size > biggest_axis_size)
            {
                axis = kk;
                biggest_axis_size = kth_axis_size;
            }
        }
        return axis;
    }

    void compute_bounding_box_of_many_boxes( int start, int stop,
                                             const vector<Box> & boxes,
                                             Box & bounding_box )
    {
        // compute limits of big box containing all boxes in this group
        for ( int kk=0; kk<K; ++kk )
        {
            double best_min_k = boxes[start].min(kk);
            for ( int bb=start+1; bb<stop; ++bb)
            {
                double candidate_min_k = boxes[bb].min(kk);
                if (candidate_min_k < best_min_k)
                {
                    best_min_k = candidate_min_k;
                }
            }
            bounding_box.min(kk) = best_min_k;
        }

        for ( int kk=0; kk<K; ++kk )
        {
            double best_max_k = boxes[start].max(kk);
            for ( int bb=start+1; bb<stop; ++bb)
            {
                double candidate_max_k = boxes[bb].max(kk);
                if ( candidate_max_k > best_max_k )
                {
                    best_max_k = candidate_max_k;
                }
            }
            bounding_box.max(kk) = best_max_k;
        }
    }

    // creates subtree and returns the index for root of subtree
    int make_subtree( int start, int stop,
                      vector<Box> & leaf_boxes,
                      int & counter ) {
        int num_boxes_local = stop - start;

        int current_node_ind = counter;
        counter = counter + 1;

        if ( num_boxes_local == 1 )
        {
            nodes[current_node_ind] = Node { leaf_boxes[start], -1, -1 };
        }
        else if (num_boxes_local > 1)
        {
            Box big_box; // bounding box for all leaf boxes in this group
            big_box.index = -1; // -1 indicates internal node
            compute_bounding_box_of_many_boxes( start, stop, leaf_boxes, big_box );

            int axis = biggest_axis_of_box( big_box );

            // Sort leaf boxes by centerpoint along biggest axis
            sort( leaf_boxes.begin() + start,
                  leaf_boxes.begin() + stop,
                  [&](Box A, Box B) {return (box_center(A, axis) > box_center(B, axis));} );

            // Find index of first leaf box with centerpoint in the "right" half of big box
            double big_box_centerpoint = box_center(big_box, axis);
            int mid = stop;
            for ( int bb=start; bb<stop; ++bb )
            {
                if ( big_box_centerpoint < box_center( leaf_boxes[bb], axis) )
                {
                    mid = bb;
                    break;
                }
            }

            // If all boxes happen to be on one side, split them in equal numbers
            // (theoretically possible, e.g., if all boxes have the same centerpoint)
            if ( (mid == start) || (mid == stop) )
            {
                mid = start + (num_boxes_local / 2);
            }

            int left  = make_subtree(start,  mid, leaf_boxes, counter);
            int right = make_subtree(mid,   stop, leaf_boxes, counter);

            nodes[current_node_ind] = Node { big_box, left, right };
            }
            else
            {
                cout << "BAD: trying to make subtree of leaf box!"
                     << " start=" << start << " stop=" << stop << " counter=" << counter
                     << endl;
            }
        return current_node_ind;
        }

    inline bool point_is_in_box( const KDVector & p, const Box & B )
    {
        bool p_is_in_box = true;
        for ( int kk=0; kk<K; ++kk)
        {
            if ( (p(kk) < B.min(kk)) || (B.max(kk) < p(kk)) )
            {
                p_is_in_box = false;
                break;
            }
        }
        return p_is_in_box;
    }

    inline bool box_is_leaf( const Box & B )
    {
        return (B.index >= 0);
    }

    int first_point_intersection_iterative( const KDVector & query )
    {
        int first_intersection = -1;

        // Normal iterative traversal of a tree uses a list.
        // However, I found that std::list is crazy slow.
        // Here I use a pre-allocated vector<int> to simulate a list. It is 2-3x faster.
        int ii = 0; // <-- This is the "pointer" to the front of the list
        nodes_under_consideration[ii] = 0; // <-- This is the "list" of ints.
        while ( ii >= 0 )
        {
            int current_node_ind =  nodes_under_consideration[ii];
            ii = ii - 1;

            Node & current_node = nodes[current_node_ind];
            Box & B = current_node.box;

            // Determine if query point is in current box
            bool query_is_in_box = true;
            for ( int kk=0; kk<K; ++kk)
            {
                if ( (query(kk) < B.min(kk)) || (B.max(kk) < query(kk)) )
                {
                    query_is_in_box = false;
                    break;
                }
            }

            if ( query_is_in_box )
            {
                if ( B.index >= 0 ) // if current box is leaf
                {
                    first_intersection = B.index;
                    break;
                }
                else // current box is internal node
                {
                    nodes_under_consideration[ii+1] = current_node.right;
                    nodes_under_consideration[ii+2] = current_node.left;
                    ii = ii + 2;
                }
            }
        }
        return first_intersection;
    }

    // finds nearest neighbor of query in subtree
    int first_point_intersection_subtree( const KDVector & query,
                                          int              current )
    {
        Node current_node = nodes[current];
        Box B = current_node.box;

        int first_intersection = -1;
        if ( point_is_in_box( query, B) )
        {
            if ( box_is_leaf( B ) )
            {
                first_intersection = B.index;
            } else { // current box is internal node
                first_intersection = first_point_intersection_subtree( query, current_node.left );
                bool no_intersection_in_left_subtree = (first_intersection < 0);
                if ( no_intersection_in_left_subtree )
                {
                    first_intersection = first_point_intersection_subtree( query, current_node.right );
                }
            }
        }

        return first_intersection;
    }

public:
    AABBTree( ) {}

    AABBTree( const Ref<const Array<double, Dynamic, K>> box_mins,
              const Ref<const Array<double, Dynamic, K>> box_maxes )
    {
        int num_leaf_boxes = box_mins.rows();

        // Copy eigen matrix input into std::vector of Boxes which will be re-ordered
        vector< Box > leaf_boxes(num_leaf_boxes);
        for ( int ii=0; ii<num_leaf_boxes; ++ii)
        {
            leaf_boxes[ii] = Box { box_mins.row(ii), box_maxes.row(ii), ii };
        }

        int num_boxes = 2*num_leaf_boxes - 1; // full binary tree with n leafs has 2n-1 nodes
        nodes_under_consideration.reserve(num_boxes);
        nodes.reserve(num_boxes);
        int counter = 0;
        int zero = make_subtree(0, num_leaf_boxes, leaf_boxes, counter);
    }

    inline int first_point_intersection( const KDVector & query )
    {
        return first_point_intersection_iterative( query );
//        return first_point_intersection_subtree( query, 0 );
    }

    VectorXi first_point_intersection_vectorized( const Ref<const Matrix<double, Dynamic, K>> query_array )
    {
        int num_querys = query_array.rows();
        VectorXi first_intersection_inds(num_querys);
        for (int ii=0; ii<num_querys; ++ii)
        {
            KDVector query = query_array.row(ii);
//            first_intersection_inds(ii) = first_point_intersection_subtree( query, 0 );
            first_intersection_inds(ii) = first_point_intersection_iterative( query );
        }
        return first_intersection_inds;
    }

//    vector<int> all_point_intersections( const KDVector & query )
    vector<int> all_point_intersections( const KDVector & query )
    {
        vector<int> all_intersections;

        int ii = 0; // <-- This is the "pointer" to the front of the list
        nodes_under_consideration[ii] = 0; // <-- This is the "list" of ints.
        while ( ii >= 0 )
        {
            int current_node_ind =  nodes_under_consideration[ii];
            ii = ii - 1;

            Node & current_node = nodes[current_node_ind];
            Box & B = current_node.box;

            // Determine if query point is in current box
            bool query_is_in_box = true;
            for ( int kk=0; kk<K; ++kk)
            {
                if ( (query(kk) < B.min(kk)) || (B.max(kk) < query(kk)) )
                {
                    query_is_in_box = false;
                    break;
                }
            }

            if ( query_is_in_box )
            {
                if ( B.index >= 0 ) // if current box is leaf
                {
                    all_intersections.push_back( B.index );
                }
                else // current box is internal node
                {
                    nodes_under_consideration[ii+1] = current_node.right;
                    nodes_under_consideration[ii+2] = current_node.left;
                    ii = ii + 2;
                }
            }
        }
        return all_intersections;
    }

    vector<int> all_ball_intersections( const KDVector & center, double radius )
    {
        vector<int> all_intersections;

        int ii = 0; // <-- This is the "pointer" to the front of the list
        nodes_under_consideration[ii] = 0; // <-- This is the "list" of ints.
        while ( ii >= 0 )
        {
            int current_node_ind =  nodes_under_consideration[ii];
            ii = ii - 1;

            Node & current_node = nodes[current_node_ind];
            Box & B = current_node.box;

            // Construct point on box that is closest to ball center
            KDVector closest_point;
            for ( int kk=0; kk<K; ++kk)
            {
                if ( center(kk) < B.min(kk) )
                {
                    closest_point(kk) = B.min(kk);
                }
                else if ( B.max(kk) < center(kk) )
                {
                    closest_point(kk) = B.max(kk);
                }
                else
                {
                    closest_point(kk) = center(kk);
                }
            }

            double distance_to_box_squared = (closest_point - center).squaredNorm();
            bool ball_intersects_box = (distance_to_box_squared <= radius*radius);

            if ( ball_intersects_box )
            {
                if ( B.index >= 0 ) // if current box is leaf
                {
                    all_intersections.push_back( B.index );
                }
                else // current box is internal node
                {
                    nodes_under_consideration[ii+1] = current_node.right;
                    nodes_under_consideration[ii+2] = current_node.left;
                    ii = ii + 2;
                }
            }
        }
        return all_intersections;
    }

};

