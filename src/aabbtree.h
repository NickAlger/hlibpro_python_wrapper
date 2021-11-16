#pragma once

#include <iostream>
#include <list>
#include <vector>
#include <queue>

#include <math.h>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;


int biggest_axis_of_box( const VectorXd box_min, const VectorXd box_max )
{
    int axis = 0;
    int dim = box_min.size();
    double biggest_axis_size = box_max(0) - box_min(0);
    for ( int kk=1; kk<dim; ++kk)
    {
        double kth_axis_size = box_max(kk) - box_min(kk);
        if (kth_axis_size > biggest_axis_size)
        {
            axis = kk;
            biggest_axis_size = kth_axis_size;
        }
    }
    return axis;
}

pair<VectorXd, VectorXd> bounding_box_of_boxes( const MatrixXd box_mins, const MatrixXd box_maxes )
{
    int dim = box_mins.rows();
    int num_boxes = box_mins.cols();
    VectorXd big_box_min = box_mins.col(0);
    VectorXd big_box_max = box_maxes.col(0);
    for ( int bb=1; bb<num_boxes; ++bb )
    {
        for ( int kk=0; kk<dim; ++kk )
        {
            double x_min = box_mins(kk, bb);
            if ( x_min < big_box_min(kk) )
            {
                big_box_min(kk) = x_min;
            }

            double x_max = box_maxes(kk, bb);
            if ( big_box_max(kk) < x_max )
            {
                big_box_max(kk) = x_max;
            }
        }
    }
    return make_pair(big_box_min, big_box_max);
}

inline unsigned int power_of_two_floor(unsigned int x)
// based on https://stackoverflow.com/a/42595922/484944
{
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return x ^ (x >> 1);
}

int power_of_two( int k ) // x=2^k, with 2^0 = 1. Why does c++ not have this!?
{
    int x = 1;
    for ( int ii=1; ii<=k; ++ii )
    {
        x = 2*x;
    }
    return x;
}

inline unsigned int heap_left_size(unsigned int N)
{
    int N_full = power_of_two_floor(N);
    int N_full_left = N_full / 2;
    int N_extra = N - N_full;
    int N_left_max = N_full;
    int N_left;
    if ( N_full_left + N_extra < N_left_max )
    {
        N_left = N_full_left + N_extra;
    }
    else
    {
        N_left = N_left_max;
    }
    return N_left;
}

class AABBTree {
private:
    int                dim;
    int                num_boxes;
    int                num_leaf_boxes;
    VectorXi           i2e;
    MatrixXd           box_mins;
    MatrixXd           box_maxes;

public:
    int block_size = 32;

    AABBTree( ) {}

    AABBTree( const Ref<const MatrixXd> input_box_mins,
              const Ref<const MatrixXd> input_box_maxes )
    {
        dim            = input_box_mins.rows();
        num_leaf_boxes = input_box_mins.cols();

        vector<int> working_i2e(num_leaf_boxes);
        iota(working_i2e.begin(), working_i2e.end(), 0);

        num_boxes = 2*num_leaf_boxes - 1; // full binary tree with n leafs has 2n-1 nodes
        i2e.resize(num_boxes);
        i2e.setConstant(-1); // -1 for internal nodes. leaf nodes will be set with indices

        box_mins.resize(dim, num_boxes);
        box_maxes.resize(dim, num_boxes);

        int counter = 0;
        queue<pair<int,int>> start_stop_candidates;
        start_stop_candidates.push(make_pair(0,num_leaf_boxes));
        while ( !start_stop_candidates.empty() )
        {
            pair<int,int> candidate = start_stop_candidates.front();
            start_stop_candidates.pop();

            int start = candidate.first;
            int stop = candidate.second;

            MatrixXd local_box_mins(dim, stop-start);
            MatrixXd local_box_maxes(dim, stop-start);
            for ( int ii=0; ii<stop-start; ++ii )
            {
                local_box_mins.col(ii) = input_box_mins.col(working_i2e[start+ii]);
                local_box_maxes.col(ii) = input_box_maxes.col(working_i2e[start+ii]);
            }
            pair<VectorXd, VectorXd> BB = bounding_box_of_boxes(local_box_mins, local_box_maxes);
            VectorXd big_box_min = BB.first;
            VectorXd big_box_max = BB.second;

            box_mins.col(counter) = big_box_min;
            box_maxes.col(counter) = big_box_max;

            if ( stop - start == 1 )
            {
                i2e(counter) = working_i2e[start];
            }
            else if ( stop - start >= 2 )
            {
                int axis = biggest_axis_of_box( big_box_min, big_box_max );

                sort( working_i2e.begin() + start,
                      working_i2e.begin() + stop,
                      [&](int aa, int bb) {return 0.5*(input_box_maxes(axis,aa)+input_box_mins(axis,aa))
                                                > 0.5*(input_box_maxes(axis,bb)+input_box_mins(axis,bb));} );

                int mid = start + heap_left_size( stop - start );
                start_stop_candidates.push(make_pair(start, mid));
                start_stop_candidates.push(make_pair(mid,   stop));
            }

            counter += 1;
        }
    }

    VectorXi point_collisions( const VectorXd & query ) const
    {
//        queue<int> boxes_under_consideration;
//        boxes_under_consideration.push(0);
        vector<int> boxes_under_consideration;
        boxes_under_consideration.reserve(100);
        boxes_under_consideration.push_back(0);

        vector<int> collision_leafs;
        collision_leafs.reserve(100);

        while ( !boxes_under_consideration.empty() )
        {
//            int B = boxes_under_consideration.front();
//            boxes_under_consideration.pop();
            int B = boxes_under_consideration.back();
            boxes_under_consideration.pop_back();

            int B_level = power_of_two_floor(B+1);
            int last_level = power_of_two_floor(num_boxes);

            int jump2 = (last_level) / (B_level);
            int jump1 = jump2 / 2;

            int start1 = (B+1)*jump1 - 1;
            int stop1  = (B+1)*jump1 - 1 + jump1;

            int start2 = (B+1)*jump2 - 1;
            int stop2  = (B+1)*jump2 - 1 + jump2;

            if ( stop2 <= num_boxes ) // All leaf descendends are on last level
            {
                start1 = stop1;
            }
            else if ( start2 >= num_boxes ) // All leaf descendends are on second to last level
            {
                start2 = stop2;
            }
            else
            {
                start1 = (num_boxes-1) / 2;
                stop2 = num_boxes;
            }

            int num_leaf_descendents = (stop2-start2) + (stop1-start1);

//            cout << "B=" << B << ", num_boxes=" << num_boxes << ", num_leaf_descendents=" << num_leaf_descendents << endl
//                 << "B_level=" << B_level << ", last_level=" << last_level << endl
//                 << "jump1=" << jump1 << ", jump2=" << jump2 << endl
//                 << "start1=" << start1 << ", stop1=" << stop1 << endl
//                 << "start2=" << start2 << ", stop2=" << stop2 << endl << endl;

            if ( num_leaf_descendents <= block_size )
            {
                if ( stop1 > start1 )
                {
                    Matrix<bool,Dynamic,1> queries_are_in_box =
                        (box_mins .middleCols(start1, stop1-start1).array().colwise() - query.array() <= 0.0 ).colwise().all() &&
                        (box_maxes.middleCols(start1, stop1-start1).array().colwise() - query.array() >= 0.0 ).colwise().all();
                    for ( int ii=0; ii<stop1-start1; ++ii )
                    {
                        if ( queries_are_in_box(ii) )
                        {
                            collision_leafs.push_back(start1 + ii);
                        }
                    }
                }

                if ( stop2 > start2 )
                {
                    Matrix<bool,Dynamic,1> queries_are_in_box =
                        (box_mins .middleCols(start2, stop2-start2).array().colwise() - query.array() <= 0.0 ).colwise().all() &&
                        (box_maxes.middleCols(start2, stop2-start2).array().colwise() - query.array() >= 0.0 ).colwise().all();
                    for ( int ii=0; ii<stop2-start2; ++ii )
                    {
                        if ( queries_are_in_box(ii) )
                        {
                            collision_leafs.push_back(start2 + ii);
                        }
                    }
                }
            }
            else
            {
                bool query_is_in_box = (box_mins .col(B).array() <= query.array()).all() &&
                                       (box_maxes.col(B).array() >= query.array()).all();

                if ( query_is_in_box )
                {
                    if ( 2*B + 1 >= num_boxes ) // if current box is leaf
                    {
                        collision_leafs.push_back(B);
                    }
                    else // current box is internal node
                    {
                        boxes_under_consideration.push_back(2*B + 1);
                        boxes_under_consideration.push_back(2*B + 2);
//                        boxes_under_consideration.push(2*B + 1);
//                        boxes_under_consideration.push(2*B + 2);
                    }
                }
            }
        }

        VectorXi collision_leafs_external_indexing(collision_leafs.size());
        for ( int ii=0; ii<collision_leafs.size(); ++ii )
        {
            collision_leafs_external_indexing(ii) = i2e(collision_leafs[ii]);
        }
        return collision_leafs_external_indexing;
    }

    VectorXi ball_collisions( const VectorXd & center, double radius ) const
    {
        double radius_squared = radius*radius;

        queue<int> boxes_under_consideration;
        boxes_under_consideration.push(0);

        vector<int> collision_leafs;
        collision_leafs.reserve(100);

        while ( !boxes_under_consideration.empty() )
        {
            int B = boxes_under_consideration.front();
            boxes_under_consideration.pop();

            // Construct point on box that is closest to ball center
            VectorXd closest_point = center.cwiseMin(box_maxes.col(B)).cwiseMax(box_mins.col(B));

            double distance_to_box_squared = (closest_point - center).squaredNorm();
            bool ball_intersects_box = (distance_to_box_squared <= radius_squared);

            if ( ball_intersects_box )
            {
                if ( 2*B + 1 >= num_boxes ) // if current box is leaf
                {
                    collision_leafs.push_back(B);
                }
                else // current box is internal node
                {
                    boxes_under_consideration.push(2*B + 1);
                    boxes_under_consideration.push(2*B + 2);
                }
            }
        }

        VectorXi collision_leafs_external_indexing(collision_leafs.size());
        for ( int ii=0; ii<collision_leafs.size(); ++ii )
        {
            collision_leafs_external_indexing(ii) = i2e(collision_leafs[ii]);
        }
        return collision_leafs_external_indexing;
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

