#pragma once

#include <iostream>
#include <stdexcept>

#include "thread-pool-master/thread_pool.hpp"

#include <math.h>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;


void geometric_sort_helper( int start,
                            int stop,
                            int depth,
                            int dim,
                            const MatrixXd & points,
                            vector<int> & sort_inds )
{
    int num_pts_local = stop - start;
    if (num_pts_local >= 2)
    {
        int axis = depth % dim;
        sort( sort_inds.begin() + start, sort_inds.begin() + stop,
              [&](int ii, int jj) {return points(axis, ii) > points(axis, jj);} );

        int mid = start + (num_pts_local / 2);
        geometric_sort_helper(start, mid,  depth + 1, dim, points, sort_inds);
        geometric_sort_helper(mid,   stop, depth + 1, dim, points, sort_inds);
    }
}


vector<int> geometric_sort( const MatrixXd & points )
{
    int dim = points.rows();
    int num_pts = points.cols();

    vector<int> sort_inds(num_pts);
    iota(sort_inds.begin(), sort_inds.end(), 0);

    geometric_sort_helper( 0, num_pts, 0, dim, points, sort_inds );

    return sort_inds;
}
