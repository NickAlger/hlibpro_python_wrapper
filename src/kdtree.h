#pragma once

#include <iostream>
#include <list>
#include <queue>
#include <vector>

#include <math.h>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;


class KDTree
{
private:
    int            num_pts;
    int            dim; // spatial dimension
    MatrixXd       points; // points_array.col(ii) is ith point (in internal ordering)
    VectorXd       splitting_coords; // coordinates of midpoints along chosen axes
    VectorXi       perm_i2e; // permutation from internal ordering to external ordering

    // creates subtree and returns the index for root of subtree
    void make_subtree( int           start,
                       int           stop,
                       int           depth,
                       const Ref<const MatrixXd> input_points,
                       vector<int> &             working_perm_i2e )
    {
        int num_pts_local = stop - start;
        if ( num_pts_local > 0 )
        {
            int axis = depth % dim;
            int mid = start + (num_pts_local / 2);

            sort( working_perm_i2e.begin() + start, working_perm_i2e.begin() + stop,
                [&axis,&input_points](int ii, int jj) {return input_points(axis,ii) < input_points(axis,jj);} );

            splitting_coords(mid) = input_points(axis, working_perm_i2e[mid]);

            if ( num_pts_local > 1 )
            {
                make_subtree(start,  mid,  depth+1, input_points, working_perm_i2e);
                make_subtree(mid+1,  stop, depth+1, input_points, working_perm_i2e);
            }

        }
    }

    // finds num_neighbors nearest neighbors of query in subtree
    void query_subtree( const VectorXd &                         query_point,
                        vector<int> &                            visited_inds,
                        vector<double> &                         visited_distances,
                        priority_queue<double, vector<double>> & best_squared_distances,
                        int                                      start,
                        int                                      stop,
                        int                                      depth,
                        int                                      num_neighbors ) const
    {
        if ( stop - start <= block_size )
        {
            VectorXd dsqs = (points.middleCols(start, stop - start).colwise() - query_point).colwise().squaredNorm();
            for ( int ii=0; ii<dsqs.size(); ++ii )
            {
                int ind = start + ii;
                double dsq = dsqs(ii);
                if ( best_squared_distances.size() < num_neighbors )
                {
                    visited_inds.push_back( ind );
                    visited_distances.push_back( dsq );
                    best_squared_distances.push( dsq );
                }
                else if ( dsq < best_squared_distances.top() )
                {
                    visited_inds.push_back( ind );
                    visited_distances.push_back( dsq );
                    best_squared_distances.pop();
                    best_squared_distances.push( dsq );
                }
            }
        }
        else
        {
            int mid = start + ((stop - start) / 2);
            int axis = depth % dim;
            double d_splitting_plane = query_point(axis) - splitting_coords(mid);

            int A_start;
            int A_stop;

            int B_start;
            int B_stop;
            if (d_splitting_plane <= 0)
            {
                A_start = start;
                A_stop = mid;

                B_start = mid+1;
                B_stop = stop;
            }
            else
            {
                B_start = start;
                B_stop = mid;

                A_start = mid+1;
                A_stop = stop;
            }

            if ( A_stop > A_start )
            {
                query_subtree( query_point, visited_inds, visited_distances, best_squared_distances, A_start, A_stop, depth+1, num_neighbors );
            }

            double dsquared_splitting_plane = d_splitting_plane*d_splitting_plane;

            if ( (dsquared_splitting_plane <= best_squared_distances.top()) || (best_squared_distances.size() < num_neighbors) )
            {
                double dsq = (points.col(mid) - query_point).squaredNorm();
                if ( best_squared_distances.size() < num_neighbors )
                {
                    visited_inds.push_back( mid );
                    visited_distances.push_back( dsq );
                    best_squared_distances.push( dsq );
                }
                else if ( dsq < best_squared_distances.top() )
                {
                    visited_inds.push_back( mid );
                    visited_distances.push_back( dsq );
                    best_squared_distances.pop();
                    best_squared_distances.push( dsq );
                }
            }

            if ( B_stop > B_start )
            {
                if ( (dsquared_splitting_plane <= best_squared_distances.top()) || (best_squared_distances.size() < num_neighbors) )
                {
                    query_subtree( query_point, visited_inds, visited_distances, best_squared_distances, B_start, B_stop, depth+1, num_neighbors );
                }
            }
        }
    }

public:
    int block_size = 32;

    KDTree( ) {}

    KDTree( const Ref<const MatrixXd> input_points )
    {
        dim = input_points.rows();
        num_pts = input_points.cols();

        vector<int> working_perm_i2e(num_pts);
        iota(working_perm_i2e.begin(), working_perm_i2e.end(), 0);

        splitting_coords.resize(num_pts);

        make_subtree(0, num_pts, 0, input_points, working_perm_i2e);

        points.resize(dim, num_pts);
        perm_i2e.resize(num_pts);
        for ( int ii=0; ii<num_pts; ++ii )
        {
            perm_i2e(ii) = working_perm_i2e[ii];
            points.col(ii) = input_points.col(working_perm_i2e[ii]);
        }
    }

    pair<MatrixXi, MatrixXd> query( const Ref<const MatrixXd> query_points, int num_neighbors ) const
    {
        int num_queries = query_points.cols();

        MatrixXi closest_point_inds(num_neighbors, num_queries);
        MatrixXd squared_distances(num_neighbors, num_queries);

        for ( int ii=0; ii<num_queries; ++ii )
        {
            vector<int> visited_inds;
            visited_inds.reserve(10*num_neighbors);

            vector<double> visited_distances;
            visited_distances.reserve(10*num_neighbors);

            vector<double> container;
            container.reserve(2*num_neighbors);
            priority_queue<double, vector<double>> best_squared_distances(less<double>(), move(container));

            query_subtree( query_points.col(ii), visited_inds, visited_distances, best_squared_distances, 0, num_pts, 0, num_neighbors );

            vector<int> sort_inds(visited_distances.size());
            iota(sort_inds.begin(), sort_inds.end(), 0);
            sort(sort_inds.begin(), sort_inds.end(),
                 [&](int aa, int bb){return visited_distances[aa] < visited_distances[bb];});

            for ( int jj=0; jj<num_neighbors; ++jj )
            {
                closest_point_inds(jj,ii) = perm_i2e(visited_inds[sort_inds[jj]]);
                squared_distances(jj,ii) = visited_distances[sort_inds[jj]];
            }
        }
        return make_pair(closest_point_inds, squared_distances);
    }

};

