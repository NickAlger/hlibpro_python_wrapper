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

struct KDNode
{
    int axis;
    double coord_along_axis;
    int left;
    int right;
};


class KDTree
{
private:
    int            num_pts;
    int            dim; // spatial dimension
    int            root;
    MatrixXd       points; // points_array.col(ii) is ith point (in internal ordering)
    vector<KDNode> nodes;
    VectorXi       perm_i2e; // permutation from internal ordering to external ordering

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
//            VectorXd cluster_min = input_points.col(working_perm_i2e[start]);
//            VectorXd cluster_max = input_points.col(working_perm_i2e[start]);
//            for ( int ind=start+1; ind<stop; ++ind )
//            {
//                for ( int kk=0; kk<dim; ++kk )
//                {
//                    double x = input_points(kk, working_perm_i2e[ind]);
//                    if ( x < cluster_min(kk) )
//                    {
//                        cluster_min(kk) = x;
//                    }
//                    else if ( cluster_max(kk) < x )
//                    {
//                        cluster_max(kk) = x;
//                    }
//                }
//            }
//
//            int axis=0;
//            double biggest_width = cluster_max(0) - cluster_min(0);
//            for ( int kk=1; kk<dim; ++kk )
//            {
//                if ( biggest_width < cluster_max(kk) - cluster_min(kk) )
//                {
//                    axis = kk;
//                }
//            }

            int axis = depth % dim;

            sort( working_perm_i2e.begin() + start, working_perm_i2e.begin() + stop,
                  [&axis,&input_points](int ii, int jj) {return input_points(axis,ii) > input_points(axis,jj);} );

            mid = start + (num_pts_local / 2);

            double coord_along_axis = input_points(axis,working_perm_i2e[mid]);

            int left = make_subtree(start,  mid, depth+1, input_points, working_perm_i2e);
            int right = make_subtree(mid+1, stop, depth+1, input_points, working_perm_i2e);

            nodes[mid] = KDNode { axis, coord_along_axis, left, right };
        }
        return mid;
    }

    // finds num_neighbors nearest neighbors of query in subtree
    void query_subtree( const VectorXd &                                       query_point,
                        priority_queue<SubtreeResult, vector<SubtreeResult>> & nn,
                        int                                                    cur_index,
                        int                                                    num_neighbors ) const
    {
        const KDNode & cur_node = nodes[cur_index];
        double displacement_to_splitting_plane = query_point(cur_node.axis) - cur_node.coord_along_axis;

        int A;
        int B;
        if (displacement_to_splitting_plane >= 0)
        {
            A = cur_node.left;
            B = cur_node.right;
        }
        else
        {
            A = cur_node.right;
            B = cur_node.left;
        }

        if (A >= 0)
        {
            query_subtree( query_point, nn, A, num_neighbors );
        }

        double dsquared_splitting_plane = displacement_to_splitting_plane*displacement_to_splitting_plane;

        if ( nn.size() < num_neighbors )
        {
            double dsq_cur = (query_point - points.col(cur_index)).squaredNorm();
            nn.push( SubtreeResult {cur_index, dsq_cur} );
        }
        else if ( dsquared_splitting_plane < nn.top().distance_squared )
        {
            double dsq_cur = (query_point - points.col(cur_index)).squaredNorm();
            if ( dsq_cur < nn.top().distance_squared )
            {
                nn.pop();
                nn.push( SubtreeResult {cur_index, dsq_cur} );
            }
        }

        if ( B >= 0 )
        {
            if ( dsquared_splitting_plane < nn.top().distance_squared )
            {
                query_subtree( query_point, nn, B, num_neighbors );
            }
        }

    }

public:
    KDTree( ) {}

    KDTree( const Ref<const MatrixXd> input_points )
    {
        dim = input_points.rows();
        num_pts = input_points.cols();

        nodes.resize(num_pts);

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

        reorder_depth_first();
    }

    void reorder_depth_first()
    {
        vector<int> order_inds;
        vector<int> working_inds;
        working_inds.push_back(root);
        while ( !working_inds.empty() )
        {
            int cur = working_inds.back();
            working_inds.pop_back();
            order_inds.push_back(cur);

            if ( nodes[cur].right >= 0 )
            {
                working_inds.push_back(nodes[cur].right);
            }
            if ( nodes[cur].left >= 0 )
            {
                working_inds.push_back(nodes[cur].left);
            }
        }

        VectorXi oi(num_pts);
        MatrixXd       points_old = points;
        VectorXi       perm_i2e_old = perm_i2e;

        VectorXi inverse_order_inds(num_pts);

        for ( int ii=0; ii<num_pts; ++ii )
        {
            oi[ii] = order_inds[ii];

            inverse_order_inds(order_inds[ii]) = ii;
            points.col(ii) = points_old.col(order_inds[ii]);
            perm_i2e(ii) = perm_i2e_old(order_inds[ii]);
        }


        vector<KDNode> nodes_old = nodes;
        for ( int ii=0; ii<num_pts; ++ii )
        {
            KDNode old_node = nodes_old[order_inds[ii]];
            int new_left = -1;
            if ( old_node.left >= 0 )
            {
                new_left = inverse_order_inds(old_node.left);
            }

            int new_right = -1;
            if ( old_node.right >= 0 )
            {
                new_right = inverse_order_inds(old_node.right);
            }

            nodes[ii] = KDNode { old_node.axis,
                                 old_node.coord_along_axis,
                                 new_left,
                                 new_right };
        }

        root = 0;
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

            query_subtree( query_points.col(ii), nn, root, num_neighbors );

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

