#pragma once

#include <iostream>
#include <list>
#include <vector>

#include <math.h>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;


struct Box { VectorXd min;
             VectorXd max;
             int                  index;}; // 0,...,N for leaf nodes, and -1 for internal node


// Node in AABB tree
struct AABBNode { Box box;
                  int left;       // index of left child
                  int right; };   // index of right child


double box_center( const Box & B, int axis )
{
    return 0.5*(B.max(axis)+B.min(axis));
}


int biggest_axis_of_box( const Box & B )
{
    int axis = 0;
    int dim = B.max.size();
    double biggest_axis_size = B.max(0) - B.min(0);
    for ( int kk=1; kk<dim; ++kk)
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
    int dim = boxes[start].max.size();
    // compute limits of big box containing all boxes in this group
    for ( int kk=0; kk<dim; ++kk )
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

    for ( int kk=0; kk<dim; ++kk )
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


class AABBTree {
private:
    int                dim;
    VectorXi           i2e;
    MatrixXd           box_mins;
    MatrixXd           box_maxes;
    vector< AABBNode > nodes; // All nodes in the tree

    // creates subtree and returns the index for root of subtree
    int make_subtree( int start, int stop,
                      vector<int> & working_i2e;
                      MatrixXd & input_box_mins;
                      MatrixXd & input_box_maxes;
                      int & counter )
    {
        int num_boxes_local = stop - start;

        int current_node_ind = counter;
        counter = counter + 1;

        if ( num_boxes_local == 1 )
        {
            nodes[current_node_ind] = AABBNode { leaf_boxes[start], -1, -1 };
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

            nodes[current_node_ind] = AABBNode { big_box, left, right };
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

    AABBTree( const Ref<const MatrixXd> input_box_mins,
              const Ref<const MatrixXd> input_box_maxes )
    {
        dim = input_box_mins.rows();
        int num_leaf_boxes = box_mins.cols();

//        // Copy eigen matrix input into std::vector of Boxes which will be re-ordered
//        vector< Box > leaf_boxes(num_leaf_boxes);
//        for ( int ii=0; ii<num_leaf_boxes; ++ii)
//        {
//            leaf_boxes[ii] = Box { input_box_mins.col(ii), input_box_maxes.col(ii), ii };
//        }

//        vector<VectorXd> leaf_mins(num_leaf_boxes);
//        vector<VectorXd> leaf_maxes(num_leaf_boxes);
//        vector<int>      leaf_i2e(num_leaf_boxes);
//        for ( int ii=0; ii<num_leaf_boxes; ++ii )
//        {
//            leaf_mins[ii] = input_box_mins.col(ii);
//            leaf_maxes[ii] = input_box_maxes.col(ii);
//            leaf_i2e[ii] = ii;
//        }

        vector<int> leaf_i2e(num_leaf_boxes);
        iota(leaf_i2e.begin(), leaf_i2e.end(), 0);


        int num_boxes = 2*num_leaf_boxes - 1; // full binary tree with n leafs has 2n-1 nodes
        nodes.reserve(num_boxes);
        int counter = 0;

        vector<tuple<int,int,int>> start_stop_axis_candidates;
        start_stop_axis_candidates.push_back(make_tuple(0,num_leaf_boxes,0));
        while ( !candidate_starts_and_stops.empty() )
        {
            tuple<int,int,int> candidate = candidate_starts_and_stops.back();
            candidate_starts_and_stops.pop_back();

            int start = get<0> candidate;
            int stop = get<1> candidate;
            int axis = get<2> candidate;

            input_box_mins.middleCols(leaf_i2e[start], stop - start).colwise()
        }


        int zero = make_subtree(0, num_leaf_boxes, leaf_boxes, counter);
    }

    VectorXi point_collisions( const VectorXd & query ) const
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

            const AABBNode & current_node = nodes[current_node_ind];
            const Box & B = current_node.box;

            // Determine if query point is in current box
            bool query_is_in_box = true;
            for ( int kk=0; kk<dim; ++kk)
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

    VectorXi ball_collisions( const VectorXd & center, double radius ) const
    {
        double radius_squared = radius*radius;

        vector<int> nodes_under_consideration;
        nodes_under_consideration.reserve(100);
        nodes_under_consideration.push_back(0);

        vector<int> collision_leafs;
        collision_leafs.reserve(100);

        while ( !nodes_under_consideration.empty() )
        {
            int current_node_ind = nodes_under_consideration.back();
            nodes_under_consideration.pop_back();

            const AABBNode & current_node = nodes[current_node_ind];
            const Box & B = current_node.box;

            // Construct point on box that is closest to ball center
            VectorXd closest_point;
            for ( int kk=0; kk<dim; ++kk)
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

    vector<VectorXi> point_collisions_vectorized( const MatrixXd & query_points) const
    {
        int num_points = query_points.cols();
        vector<VectorXi> all_collisions(num_points);
        for ( int ii=0; ii<num_points; ++ii )
        {
            all_collisions[ii] = point_collisions( query_points.col(ii) );
        }
        return all_collisions;
    }

    vector<VectorXi> ball_collisions_vectorized( const Ref<const MatrixXd> centers,
                                                 const Ref<const VectorXd>                   radii ) const
    {
        int num_balls = centers.cols();
        vector<VectorXi> all_collisions(num_balls);
        for ( int ii=0; ii<num_balls; ++ii )
        {
            all_collisions[ii] = ball_collisions( centers.col(ii), radii(ii) );
        }
        return all_collisions;
    }

};

