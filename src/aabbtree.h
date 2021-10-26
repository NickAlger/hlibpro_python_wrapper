#pragma once

#include <iostream>
#include <list>

#include <math.h>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;


template <int K>
struct Box { Matrix<double, K, 1> min;
             Matrix<double, K, 1> max;
             int                  index;}; // 0,...,N for leaf nodes, and -1 for internal node


// Node in AABB tree
template <int K>
struct AABBNode { Box<K> box;
                  int left;       // index of left child
                  int right; };   // index of right child


template <int K>
double box_center( const Box<K> & B, int axis )
{
    return 0.5*(B.max(axis)+B.min(axis));
}


template <int K>
int biggest_axis_of_box( const Box<K> & B )
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


template <int K>
void compute_bounding_box_of_many_boxes( int start, int stop,
                                         const vector<Box<K>> & boxes,
                                         Box<K> & bounding_box )
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


template <int K>
class AABBTree {
private:
    vector< AABBNode<K> > nodes; // All nodes in the tree
    VectorXi nodes_under_consideration;
    VectorXi selected_nodes;

    // creates subtree and returns the index for root of subtree
    int make_subtree( int start, int stop,
                      vector<Box<K>> & leaf_boxes,
                      int & counter )
    {
        int num_boxes_local = stop - start;

        int current_node_ind = counter;
        counter = counter + 1;

        if ( num_boxes_local == 1 )
        {
            nodes[current_node_ind] = AABBNode<K> { leaf_boxes[start], -1, -1 };
        }
        else if (num_boxes_local > 1)
        {
            Box<K> big_box; // bounding box for all leaf boxes in this group
            big_box.index = -1; // -1 indicates internal node
            compute_bounding_box_of_many_boxes<K>( start, stop, leaf_boxes, big_box );

            int axis = biggest_axis_of_box<K>( big_box );

            // Sort leaf boxes by centerpoint along biggest axis
            sort( leaf_boxes.begin() + start,
                  leaf_boxes.begin() + stop,
                  [&](Box<K> A, Box<K> B) {return (box_center<K>(A, axis) > box_center<K>(B, axis));} );

            // Find index of first leaf box with centerpoint in the "right" half of big box
            double big_box_centerpoint = box_center<K>(big_box, axis);
            int mid = stop;
            for ( int bb=start; bb<stop; ++bb )
            {
                if ( big_box_centerpoint < box_center<K>( leaf_boxes[bb], axis) )
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

            nodes[current_node_ind] = AABBNode<K> { big_box, left, right };
            }
            else
            {
                cout << "BAD: trying to make subtree of leaf box!"
                     << " start=" << start << " stop=" << stop << " counter=" << counter
                     << endl;
            }
        return current_node_ind;
    }

public:
    AABBTree( ) {}

    AABBTree( const Ref<const Array<double, K, Dynamic>> box_mins,
              const Ref<const Array<double, K, Dynamic>> box_maxes )
    {
        int num_leaf_boxes = box_mins.cols();

        // Copy eigen matrix input into std::vector of Boxes which will be re-ordered
        vector< Box<K> > leaf_boxes(num_leaf_boxes);
        for ( int ii=0; ii<num_leaf_boxes; ++ii)
        {
            leaf_boxes[ii] = Box<K> { box_mins.col(ii), box_maxes.col(ii), ii };
        }

        int num_boxes = 2*num_leaf_boxes - 1; // full binary tree with n leafs has 2n-1 nodes
        nodes_under_consideration.resize(num_boxes, 1);
        selected_nodes.resize(num_boxes, 1);
        nodes.reserve(num_boxes);
        int counter = 0;
        int zero = make_subtree(0, num_leaf_boxes, leaf_boxes, counter);
    }

    int first_point_intersection( const Matrix<double, K, 1> & query )
    {
        int first_intersection = -1;

        // Normal iterative traversal of a tree uses a list.
        // However, I found that std::list is crazy slow.
        // Here I use a pre-allocated vector<int> to simulate a list. It is 2-3x faster.
        int ii = 0; // <-- This is the "pointer" to the front of the list
        nodes_under_consideration(ii) = 0; // <-- This is the "list" of ints.
        while ( ii >= 0 )
        {
            int current_node_ind =  nodes_under_consideration(ii);
            ii = ii - 1;

            AABBNode<K> & current_node = nodes[current_node_ind];
            Box<K> & B = current_node.box;

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
                    nodes_under_consideration(ii+1) = current_node.right;
                    nodes_under_consideration(ii+2) = current_node.left;
                    ii = ii + 2;
                }
            }
        }
        return first_intersection;
    }

    VectorXi first_point_intersection_vectorized( const Ref<const Matrix<double, K, Dynamic>> query_array )
    {
        int num_querys = query_array.cols();
        VectorXi first_intersection_inds(num_querys);
        for (int ii=0; ii<num_querys; ++ii)
        {
            first_intersection_inds(ii) = first_point_intersection( query_array.col(ii) );
        }
        return first_intersection_inds;
    }

    VectorXi all_point_intersections( const Matrix<double, K, 1> & query )
    {
        vector<int> nodes_under_consideration;
        nodes_under_consideration.reserve(100);
        nodes_under_consideration.push_back(0);

        vector<int> collision_leafs;
        collision_leafs.reserve(100);

        while ( !nodes_under_consideration.empty() )
        {
            int current_node_ind = nodes_under_consideration.back();
            nodes_under_consideration.pop_back();

            AABBNode<K> & current_node = nodes[current_node_ind];
            Box<K> & B = current_node.box;

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
                    collision_leafs.push_back(B.index);
                }
                else // current box is internal node
                {
                    nodes_under_consideration.push_back(current_node.right);
                    nodes_under_consideration.push_back(current_node.left);
                }
            }
        }

        VectorXi collision_leafs_eigen(collision_leafs.size());
        for ( int ii=0; ii<collision_leafs.size(); ++ii )
        {
            collision_leafs_eigen(ii) = collision_leafs[ii];
        }
        return collision_leafs_eigen;
    }

    VectorXi all_ball_intersections( const Matrix<double, K, 1> & center, double radius )
    {
        double radius_squared = radius*radius;

        vector<int> nodes_under_consideration;
        nodes_under_consideration.reserve(100);
        nodes_under_consideration.push_back(0);

        vector<int> collision_leafs;
        collision_leafs.reserve(100);

//        int ii_consideration = 0; // <-- This is the "pointer" to the front of the list
//        nodes_under_consideration(ii_consideration) = 0; // <-- This is the "list" of ints.
//
//        int jj_selected = 0;
//        selected_nodes(jj_selected) = 0;

//        while ( ii_consideration >= 0 )
//        {
//            int current_node_ind =  nodes_under_consideration(ii_consideration);
//            ii_consideration = ii_consideration - 1;
        while ( !nodes_under_consideration.empty() )
        {
            int current_node_ind = nodes_under_consideration.back();
            nodes_under_consideration.pop_back();

            AABBNode<K> & current_node = nodes[current_node_ind];
            Box<K> & B = current_node.box;

            // Construct point on box that is closest to ball center
            Matrix<double, K, 1> closest_point;
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
            bool ball_intersects_box = (distance_to_box_squared <= radius_squared);

            if ( ball_intersects_box )
            {
                if ( B.index >= 0 ) // if current box is leaf
                {
                    collision_leafs.push_back(B.index);
//                    selected_nodes[jj_selected] = B.index;
//                    jj_selected = jj_selected + 1;
                }
                else // current box is internal node
                {
                    nodes_under_consideration.push_back(current_node.right);
                    nodes_under_consideration.push_back(current_node.left);
//                    nodes_under_consideration(ii_consideration+1) = current_node.right;
//                    nodes_under_consideration(ii_consideration+2) = current_node.left;
//                    ii_consideration = ii_consideration + 2;
                }
            }
        }

        VectorXi collision_leafs_eigen(collision_leafs.size());
        for ( int ii=0; ii<collision_leafs.size(); ++ii )
        {
            collision_leafs_eigen(ii) = collision_leafs[ii];
        }
        return collision_leafs_eigen;
//        return selected_nodes.head(jj_selected);
    }

vector<VectorXi> all_ball_intersections_vectorized( const Ref<const Matrix<double, K, Dynamic>> centers,
                                                    const Ref<const VectorXd>                   radii )
    {
        int num_balls = centers.cols();
        vector<VectorXi> all_intersections(num_balls);
        for ( int ii=0; ii<num_balls; ++ii )
        {
            all_intersections[ii] = all_ball_intersections( centers.col(ii), radii(ii) );
        }
        return all_intersections;
    }

};

