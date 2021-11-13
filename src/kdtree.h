#pragma once

#include <iostream>
#include <list>
#include <queue>
#include <vector>

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


class KDTree
{
private:
    int                   num_pts;
    int                   dim; // spatial dimension
    int                   root;
    MatrixXd              points; // points_array.col(ii) is ith point (in internal ordering)
    Matrix<int,2,Dynamic> children; // children(0,ii) is "left" child of node ii, children(1,ii) is "right" child of node ii
    VectorXi              perm_i2e; // permutation from internal ordering to external ordering

    // creates subtree and returns the index for root of subtree
    int make_subtree( int           start,
                      int           stop,
                      int           depth,
                      const Ref<const MatrixXd> input_points,
                      vector<int> &             working_perm_i2e )
    {
        int num_pts_local = stop - start;
        int mid = -1; // -1 indicates node does not exist
        if (num_pts_local >= 1)
        {
            int axis = depth % dim;
            sort( working_perm_i2e.begin() + start, working_perm_i2e.begin() + stop,
                  [&axis,&input_points](int ii, int jj) {return input_points(axis,ii) > input_points(axis,jj);} );

            mid = start + (num_pts_local / 2);

            children(0,mid) = make_subtree(start,  mid, depth+1, input_points, working_perm_i2e);
            children(1,mid) = make_subtree(mid+1, stop, depth+1, input_points, working_perm_i2e);
        }
        return mid;
    }

    // finds num_neighbors nearest neighbors of query in subtree
    void query_subtree( const VectorXd &                                       query_point,
                        priority_queue<SubtreeResult, vector<SubtreeResult>> & nn,
                        int                                                    cur_index,
                        int                                                    depth,
                        int                                                    num_neighbors ) const
    {
        const VectorXd delta = query_point - points.col(cur_index);
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

        int axis = depth % dim;
        double displacement_to_splitting_plane = delta(axis);

        int A;
        int B;
        if (displacement_to_splitting_plane >= 0)
        {
            A = children(0,cur_index);
            B = children(1,cur_index);
        }
        else
        {
            A = children(1,cur_index);
            B = children(0,cur_index);
        }

        if (A >= 0)
        {
            query_subtree( query_point, nn, A, depth+1, num_neighbors );
        }

        if (B >= 0)
        {
            if ( displacement_to_splitting_plane*displacement_to_splitting_plane
                 < nn.top().distance_squared )
            {
                query_subtree( query_point, nn, B, depth+1, num_neighbors );
            }
        }
    }

public:
    KDTree( ) {}

    KDTree( const Ref<const MatrixXd> input_points )
    {
        dim = input_points.rows();
        num_pts = input_points.cols();

        children.resize(2,num_pts);

        vector<int> working_perm_i2e(num_pts);
        iota(working_perm_i2e.begin(), working_perm_i2e.end(), 0);

        root = make_subtree(0, num_pts, 0, input_points, working_perm_i2e);

        points.resize(dim, num_pts);
        perm_i2e.resize(num_pts);
        for ( int ii=0; ii<num_pts; ++ii )
        {
            perm_i2e(ii) = working_perm_i2e[ii];
            points.col(ii) = input_points.col(working_perm_i2e[ii]);
        }
    }

    // Many queries, many neighbors each
    pair<MatrixXi, MatrixXd> query( const Ref<const MatrixXd> query_points, int num_neighbors ) const
    {
        int num_queries = query_points.cols();

        MatrixXi closest_point_inds(num_neighbors, num_queries);
        MatrixXd squared_distances(num_neighbors, num_queries);

        for ( int ii=0; ii<num_queries; ++ii )
        {
            vector<SubtreeResult> nn_container;
            nn_container.reserve(2*num_neighbors);
            priority_queue<SubtreeResult, vector<SubtreeResult>> nn(less<SubtreeResult>(), move(nn_container));

            query_subtree( query_points.col(ii), nn, root, 0, num_neighbors );

            for ( int kk=0; kk<num_neighbors; ++kk )
            {
                int jj = num_neighbors - kk - 1;
                const SubtreeResult n_kk = nn.top();
                nn.pop();
                closest_point_inds(jj,ii) = perm_i2e(n_kk.index);
                squared_distances(jj,ii) = n_kk.distance_squared;
            }
        }
        return make_pair(closest_point_inds, squared_distances);
    }

};

