#pragma once

#include <iostream>
#include <list>
#include <queue>

#include <math.h>
//#include <Eigen/Dense>

//using namespace Eigen;
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

// Node in KD tree
template <int K>
struct KDNode { array<double,K> point;
                int             left;       // index of left child
                int             right; };   // index of right child

template <int K>
struct PointWithIndex { array<double,K> point;
                        int             index; };


template <int K>
inline array<double,K> difference( const array<double,K> & p,
                                   const array<double,K> & q )
{
    array<double,K> p_minus_q;
    for ( int ii=0; ii<K; ++ii )
    {
        p_minus_q[ii] = p[ii] - q[ii];
    }
    return p_minus_q;
}

template <int K>
inline double squared_norm( const array<double,K> & p )
{
    double nsq = 0.0;
    for ( int ii=0; ii<K; ++ii )
    {
        nsq += p[ii]*p[ii];
    }
    return nsq;
}

template <int K>
class KDTree
{
private:
    vector<KDNode<K>> nodes; // All nodes in the tree
    vector<int>       perm_i2e; // permutation from internal ordering to external ordering

    // creates subtree and returns the index for root of subtree
    int make_subtree( int                         start,
                      int                         stop,
                      int                         depth,
                      vector<PointWithIndex<K>> & points,
                      int &                       counter )
    {
        int num_pts_local = stop - start;
        int current_node_ind = -1; // -1 indicates node does not exist
        if (num_pts_local >= 1)
        {
            current_node_ind = counter;
            counter = counter + 1;

            int axis = depth % K;
            sort( points.begin() + start, points.begin() + stop,
                  [axis](PointWithIndex<K> u, PointWithIndex<K> v) {return u.point[axis] > v.point[axis];} );

            int mid = start + (num_pts_local / 2);

            int left_start = start;
            int left_stop = mid;

            int right_start = mid + 1;
            int right_stop = stop;

            int left = make_subtree(left_start, left_stop, depth + 1, points, counter);
            int right = make_subtree(right_start, right_stop, depth + 1, points, counter);

            nodes[current_node_ind] = KDNode<K> { points[mid].point, left, right };
            perm_i2e[current_node_ind] = points[mid].index;
        }
        return current_node_ind;
    }

    // finds nearest neighbor of query in subtree
    SubtreeResult nn_subtree( const array<double,K> & query,
                              int                     cur_index,
                              int                     depth      ) const
    {
        const KDNode<K> & cur = nodes[cur_index];

        array<double,K> delta = difference<K>(query, cur.point);

        int best_index = cur_index;
        double best_distance_squared = squared_norm<K>(delta);

        int axis = depth % K;
        double displacement_to_splitting_plane = delta[axis];

        int A;
        int B;
        if (displacement_to_splitting_plane >= 0)
        {
            A = cur.left;
            B = cur.right;
        }
        else
        {
            A = cur.right;
            B = cur.left;
        }

        if (A >= 0)
        {
            SubtreeResult nn_A = nn_subtree( query, A, depth + 1);
            if (nn_A.distance_squared < best_distance_squared)
            {
                best_index = nn_A.index;
                best_distance_squared = nn_A.distance_squared;
            }
        }

        if (B >= 0)
        {
            bool nearest_neighbor_might_be_in_B_subtree =
                displacement_to_splitting_plane*displacement_to_splitting_plane < best_distance_squared;
            if ( nearest_neighbor_might_be_in_B_subtree )
            {
                SubtreeResult nn_B = nn_subtree( query, B, depth + 1);
                if (nn_B.distance_squared < best_distance_squared)
                {
                    best_index = nn_B.index;
                    best_distance_squared = nn_B.distance_squared;
                }
            }
        }

        return SubtreeResult { best_index, best_distance_squared };
    }

    // finds num_neighbors nearest neighbors of query in subtree
    void nn_subtree_many( const array<double,K> &                                query,
                          priority_queue<SubtreeResult, vector<SubtreeResult>> & nn,
                          int                                                    cur_index,
                          int                                                    depth,
                          int                                                    num_neighbors ) const
    {
        KDNode<K> cur = nodes[cur_index];

        const array<double,K> & delta = difference<K>(query, cur.point);
        double dsq_cur = squared_norm<K>(delta);
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

        int axis = depth % K;
        double displacement_to_splitting_plane = delta[axis];

        int A;
        int B;
        if (displacement_to_splitting_plane >= 0)
        {
            A = cur.left;
            B = cur.right;
        }
        else
        {
            A = cur.right;
            B = cur.left;
        }

        if (A >= 0)
        {
            nn_subtree_many( query, nn, A, depth+1, num_neighbors );
        }

        if (B >= 0)
        {
            if ( displacement_to_splitting_plane*displacement_to_splitting_plane
                 < nn.top().distance_squared )
            {
                nn_subtree_many( query, nn, B, depth+1, num_neighbors );
            }
        }
    }

public:
    KDTree( ) {}

    KDTree( const vector<array<double, K>> & input_points )
    {
        int num_pts = input_points.size();

        // Copy points into std::vector of tuples which will be re-ordered
        vector< PointWithIndex<K> > points(num_pts); // (coords, original_index)
        for ( int ii=0; ii<num_pts; ++ii)
        {
            points[ii].point = input_points[ii];
            points[ii].index = ii;
        }

        nodes.reserve(num_pts);
        perm_i2e.resize(num_pts, 1);
        int counter = 0;
        int zero = make_subtree(0, num_pts, 0, points, counter);
    }

    // one nearest neighbor to one point
    pair<int, double> nearest_neighbor( const array<double,K> & point ) const
    {
        SubtreeResult nn_result = nn_subtree( point, 0, 0 );
        return make_pair(perm_i2e[nn_result.index], nn_result.distance_squared);
    }

    // many nearest neighbors to one point
    pair<vector<int>, vector<double>> nearest_neighbor( const array<double,K> & point,
                                                        int                     num_neighbors ) const
    {
        vector<SubtreeResult> nn_container;
        nn_container.reserve(2*num_neighbors);
        priority_queue<SubtreeResult, vector<SubtreeResult>> nn(less<SubtreeResult>(), move(nn_container));

        nn_subtree_many( point, nn, 0, 0, num_neighbors );

        vector<int> nn_inds(num_neighbors);
        vector<double> nn_dsq(num_neighbors);
        for ( int ii=0; ii<num_neighbors; ++ii )
        {
            int jj = num_neighbors - ii - 1;
            SubtreeResult n_ii = nn.top();
            nn.pop();
            nn_inds[jj] = perm_i2e[n_ii.index];
            nn_dsq[jj] = n_ii.distance_squared;
        }
        return make_pair(nn_inds, nn_dsq);
    }

    // one nearest neighbor to many points
    pair< vector<int>, vector<double> >
        nearest_neighbor( const vector<array<double,K>> & querys ) const
    {
        int num_querys = querys.size();

        vector<int> closest_points_inds(num_querys);
        vector<double> squared_distances(num_querys);

        for ( int ii=0; ii<num_querys; ++ii )
        {
            SubtreeResult nn_result = nn_subtree( querys[ii], 0, 0 );
            closest_points_inds[ii] = perm_i2e[nn_result.index];
            squared_distances[ii] = nn_result.distance_squared;
        }

        return make_pair(closest_points_inds, squared_distances);
    }

    // many nearest neighbors to many points
    pair< vector<vector<int>>, vector<vector<double>> >
        nearest_neighbor( const vector<array<double,K>> & querys,
                          int                             num_neighbors ) const
    {
        int num_querys = querys.size();

        vector<vector<int>> closest_point_inds(num_querys);
        vector<vector<double>> squared_distances(num_querys);

        for ( int ii=0; ii<num_querys; ++ii )
        {
            pair<vector<int>, vector<double>> result = nearest_neighbor( querys[ii], num_neighbors );
            closest_point_inds[ii] = result.first;
            squared_distances[ii] = result.second;
        }
        return make_pair(closest_point_inds, squared_distances);
    }

};

