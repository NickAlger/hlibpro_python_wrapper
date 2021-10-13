#include <iostream>
#include <list>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>

#include <math.h>
#include <Eigen/Dense>

//typedef CGAL::Simple_cartesian<double> K;
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::FT FT;
typedef K::Ray_3 Ray;
typedef K::Line_3 Line;

typedef K::Point_3 Point;
typedef K::Triangle_3 Triangle;
typedef std::list<Triangle>::iterator Iterator;
typedef CGAL::AABB_triangle_primitive<K, Iterator> Primitive;
typedef CGAL::AABB_traits<K, Primitive> AABB_triangle_traits;
typedef CGAL::AABB_tree<AABB_triangle_traits> Tree;

using namespace Eigen;
using namespace std;


class AABBTreeWrapper
{
private:
    std::vector<Point> CGAL_points;
    std::vector<Triangle> TT_vector;
    std::list<Triangle> TT;

public:
    Tree aabb_tree;

    AABBTreeWrapper( Array<double, Dynamic, 2> const & points_array,
                     Array<int, Dynamic, 3> const &    triangles_array );

    Vector2d closest_point( Vector2d p );
    Array<double, Dynamic, 2> closest_points( Array<double, Dynamic, 2> const & points_array );
};

//class KDTree2D
//{
//private:
//    std::vector< std::tuple<double, double, int, int> > nodes; // (x, y, left_child_ind, right_child_ind)
//    std::vector< std::tuple<double, double> > points_vector;
//    int num_pts;
//    int dim = 2;
//    int current_node_ind = 0;
//
//    int make_kdtree(int begin_ind, int end_ind, int depth)
//    {
//        int num_pts_local = end_ind - begin_ind;
//        int mid_node_ind = -1; // -1 indicates node does not exist
//        if (num_pts_local >= 1)
//        {
//            mid_node_ind = current_node_ind;
//            current_node_ind = current_node_ind + 1;
//
//            int axis = depth % dim;
//            switch(axis)
//            {
//                case 0: std::sort(points_vector.begin() + begin_ind,
//                                  points_vector.begin() + end_ind,
//                                  [](std::tuple<double, double> u, std::tuple<double, double> v)
//                                    {return std::get<0>(u) > std::get<0>(v);} );
//                case 1: std::sort(points_vector.begin() + begin_ind,
//                                  points_vector.begin() + end_ind,
//                                  [](std::tuple<double, double> u, std::tuple<double, double> v)
//                                    {return std::get<1>(u) > std::get<1>(v);} );
//            }
////            std::sort(points_vector.begin()+begin_ind, points_vector.begin()+end_ind,
////                      [axis](Vector2d u, Vector2d v) {return u(axis) > v(axis);} );
//
//            int mid_points_ind = begin_ind + (num_pts_local / 2);
//
//            int left_begin_ind = begin_ind;
//            int left_end_ind = mid_points_ind;
//
//            int right_begin_ind = mid_points_ind + 1;
//            int right_end_ind = end_ind;
//
//            int left_node_ind = make_kdtree(left_begin_ind, left_end_ind, depth + 1);
//            int right_node_ind = make_kdtree(right_begin_ind, right_end_ind, depth + 1);
//
//            nodes[mid_node_ind] = std::make_tuple(std::get<0>(points_vector[mid_points_ind]),
//                                                  std::get<1>(points_vector[mid_points_ind]),
//                                                  left_node_ind,
//                                                  right_node_ind);
//        }
//        return mid_node_ind;
//    }
//
//    std::pair< int, double > // (node index of nearest neighbor, squared distance)
//        nearest_neighbor_subtree( double query_point_x,
//                                  double query_point_y,
//                                  int    root_index,
//                                  int    depth)
//    {
//        std::tuple<double, double, int, int> root_node = nodes[root_index];
//
//        double root_x = std::get<0>(root_node);
//        double root_y = std::get<1>(root_node);
//        int left_child_index = std::get<2>(root_node);
//        int right_child_index = std::get<3>(root_node);
//
//        double root_delta_x = query_point_x - root_x;
//        double root_delta_y = query_point_y - root_y;
//
//        int best_node_index = root_index;
//        double best_distance_squared = root_delta_x*root_delta_x + root_delta_y*root_delta_y;
//
//        int axis = depth % dim;
//        double displacement_to_splitting_plane = 0.0;
//        switch (axis)
//        {
//            case 0: displacement_to_splitting_plane = root_delta_x;
//            case 1: displacement_to_splitting_plane = root_delta_y;
//        }
//
//        int child_A_index;
//        int child_B_index;
//        if (displacement_to_splitting_plane >= 0)
//        {
//            child_A_index = left_child_index;
//            child_B_index = right_child_index;
//        } else {
//            child_A_index = right_child_index;
//            child_B_index = left_child_index;
//        }
//
//        if (child_A_index >= 0)
//        {
//            std::pair< int, double > nn_result_A =
//                nearest_neighbor_subtree( query_point_x, query_point_y, child_A_index, depth + 1);
//            int A_best_index = nn_result_A.first;
//            double A_distance_squared = nn_result_A.second;
//            if (A_distance_squared < best_distance_squared)
//            {
//                best_node_index = A_best_index;
//                best_distance_squared = A_distance_squared;
//            }
//        }
//
//        if (child_B_index >= 0)
//        {
//            if (displacement_to_splitting_plane*displacement_to_splitting_plane < best_distance_squared)
//            {
//                std::pair< int, double > nn_result_B =
//                    nearest_neighbor_subtree( query_point_x, query_point_y, child_B_index, depth + 1);
//                int B_best_index = nn_result_B.first;
//                double B_distance_squared = nn_result_B.second;
//                if (B_distance_squared < best_distance_squared)
//                {
//                    best_node_index = B_best_index;
//                    best_distance_squared = B_distance_squared;
//                }
//            }
//        }
//
//        return std::make_pair(best_node_index, best_distance_squared);
//    }
//
//public:
//    KDTree2D( Array<double, Dynamic, 2> points_array )
//    {
//        num_pts = points_array.rows();
//
//        // Copy eigen matrix input into std::vector of tuples
//        points_vector.reserve(num_pts);
//        for ( int ii=0; ii<num_pts; ++ii)
//        {
//            points_vector[ii] = std::make_tuple(points_array(ii,0), points_array(ii,1));
////            points_vector[ii] = points_array.row(ii);
//        }
//
//        nodes.reserve(num_pts);
//        int zero = make_kdtree(0, num_pts, 0);
//
////        for ( int ii=0; ii<num_pts; ++ii)
////        {
////            std::tuple<Vector2d, int, int> n = nodes[ii];
////            cout << "ii=" << ii << ", vec=" << std::get<0>(n) << endl;
////        }
//    }
//
//    std::pair<std::tuple<double, double>, double> nearest_neighbor( std::tuple<double, double> point )
//    {
//        std::pair< int, double > nn_result =
//            nearest_neighbor_subtree( std::get<0>(point),
//                                      std::get<1>(point),
//                                      0, 0);
//        int nearest_ind = nn_result.first;
//        double nearest_distance_squared = nn_result.second;
//        std::tuple<double, double, int, int> nearest_node = nodes[nearest_ind];
//        std::tuple<double, double> nearest_point = std::make_tuple(std::get<0>(nearest_node),
//                                                                   std::get<1>(nearest_node));
//        return std::make_pair(nearest_point, nearest_distance_squared);
//    }
//
//    std::pair< Array<double, Dynamic, 2>, VectorXd > nearest_neighbor_vectorized( Array<double, Dynamic, 2> query_points_array )
//    {
//        int num_query_points = query_points_array.rows();
//
//        Array<double, Dynamic, 2> closest_points_array;
//        closest_points_array.resize(num_query_points, 2);
//
//        VectorXd squared_distances(num_query_points);
//
//        for ( int ii=0; ii<num_query_points; ++ii )
//        {
//            std::tuple<double, double> qi = std::make_tuple(query_points_array(ii,0),
//                                                            query_points_array(ii,1));
//
//            std::pair<std::tuple<double, double>, double> nn_result = nearest_neighbor( qi );
//
//            std::tuple<double, double> nearest_point = nn_result.first;
//            double nearest_distance_squared = nn_result.second;
//
//            closest_points_array(ii,0) = std::get<0>(nearest_point);
//            closest_points_array(ii,1) = std::get<1>(nearest_point);
//            squared_distances(ii) = nearest_distance_squared;
//        }
//
//        return std::make_pair(closest_points_array, squared_distances);
//    }
//
//};

class KDTree2D
{
private:
    std::vector< std::tuple<Vector2d, int, int> > nodes; // (x, y, left_child_ind, right_child_ind)
    std::vector< Vector2d > points_vector;
    int num_pts;
    int dim = 2;
    int current_node_ind = 0;

    int make_kdtree(int begin_ind, int end_ind, int depth)
    {
        int num_pts_local = end_ind - begin_ind;
        int mid_node_ind = -1; // -1 indicates node does not exist
        if (num_pts_local >= 1)
        {
            mid_node_ind = current_node_ind;
            current_node_ind = current_node_ind + 1;

            int axis = depth % dim;
            std::sort(points_vector.begin() + begin_ind,
                      points_vector.begin() + end_ind,
                      [axis](Vector2d u, Vector2d v)
                                    {return u(axis) > v(axis);} );

            int mid_points_ind = begin_ind + (num_pts_local / 2);

            int left_begin_ind = begin_ind;
            int left_end_ind = mid_points_ind;

            int right_begin_ind = mid_points_ind + 1;
            int right_end_ind = end_ind;

            int left_node_ind = make_kdtree(left_begin_ind, left_end_ind, depth + 1);
            int right_node_ind = make_kdtree(right_begin_ind, right_end_ind, depth + 1);

            nodes[mid_node_ind] = std::make_tuple(points_vector[mid_points_ind],
                                                  left_node_ind,
                                                  right_node_ind);
        }
        return mid_node_ind;
    }

    std::pair< int, double > // (node index of nearest neighbor, squared distance)
        nearest_neighbor_subtree( Vector2d query_point,
                                  int    root_index,
                                  int    depth)
    {
        std::tuple<Vector2d, int, int> root_node = nodes[root_index];

        Vector2d root_vector = std::get<0>(root_node);
        int left_child_index = std::get<1>(root_node);
        int right_child_index = std::get<2>(root_node);

        Vector2d root_delta = query_point - root_vector;

        int best_node_index = root_index;
        double best_distance_squared = root_delta.squaredNorm();

        int axis = depth % dim;
        double displacement_to_splitting_plane = root_delta(axis);

        int child_A_index;
        int child_B_index;
        if (displacement_to_splitting_plane >= 0)
        {
            child_A_index = left_child_index;
            child_B_index = right_child_index;
        } else {
            child_A_index = right_child_index;
            child_B_index = left_child_index;
        }

        if (child_A_index >= 0)
        {
            std::pair< int, double > nn_result_A =
                nearest_neighbor_subtree( query_point, child_A_index, depth + 1);
            int A_best_index = nn_result_A.first;
            double A_distance_squared = nn_result_A.second;
            if (A_distance_squared < best_distance_squared)
            {
                best_node_index = A_best_index;
                best_distance_squared = A_distance_squared;
            }
        }

        if (child_B_index >= 0)
        {
            if (displacement_to_splitting_plane*displacement_to_splitting_plane < best_distance_squared)
            {
                std::pair< int, double > nn_result_B =
                    nearest_neighbor_subtree( query_point, child_B_index, depth + 1);
                int B_best_index = nn_result_B.first;
                double B_distance_squared = nn_result_B.second;
                if (B_distance_squared < best_distance_squared)
                {
                    best_node_index = B_best_index;
                    best_distance_squared = B_distance_squared;
                }
            }
        }

        return std::make_pair(best_node_index, best_distance_squared);
    }

public:
    KDTree2D( Array<double, Dynamic, 2> points_array )
    {
        num_pts = points_array.rows();

        // Copy eigen matrix input into std::vector of tuples
        points_vector.reserve(num_pts);
        for ( int ii=0; ii<num_pts; ++ii)
        {
            points_vector[ii] = points_array.row(ii);
        }

        nodes.reserve(num_pts);
        int zero = make_kdtree(0, num_pts, 0);
    }

    std::pair<Vector2d, double> nearest_neighbor( Vector2d point )
    {
        std::pair< int, double > nn_result = nearest_neighbor_subtree( point, 0, 0);
        int nearest_ind = nn_result.first;
        double nearest_distance_squared = nn_result.second;
        Vector2d nearest_point = std::get<0>(nodes[nearest_ind]);
        return std::make_pair(nearest_point, nearest_distance_squared);
    }

    std::pair< Array<double, Dynamic, 2>, VectorXd > nearest_neighbor_vectorized( Array<double, Dynamic, 2> query_points_array )
    {
        int num_query_points = query_points_array.rows();

        Array<double, Dynamic, 2> closest_points_array;
        closest_points_array.resize(num_query_points, 2);

        VectorXd squared_distances(num_query_points);

        for ( int ii=0; ii<num_query_points; ++ii )
        {
            std::pair<Vector2d, double> nn_result = nearest_neighbor( query_points_array.row(ii) );
            closest_points_array.row(ii) = nn_result.first;
            squared_distances(ii) = nn_result.second;
        }

        return std::make_pair(closest_points_array, squared_distances);
    }

};


/*
import numpy as np
import dolfin as dl
import hlibpro_python_wrapper as hpro

hcpp = hpro.hpro_cpp
mesh = dl.UnitSquareMesh(3,4)
pp = mesh.coordinates()

KDT = hcpp.KDTree2D(pp)

pp = np.random.randn(100,2)
KDT = hcpp.KDTree2D(pp)

q = np.random.randn(2)

# nearest_point, dsq = KDT.nearest_neighbor(q)
#
# nearest_ind = np.argmin(np.linalg.norm(pp - q, axis=1))
# nearest_point_true = pp[nearest_ind, :]
# dsq_true = np.linalg.norm(nearest_point_true - q)**2

qq = np.random.randn(77, 2)
nearest_points, dsqq = KDT.nearest_neighbor_vectorized(qq)

nearest_inds = np.argmin(np.linalg.norm(pp[:,None,:] - qq[None,:,:], axis=2), axis=0)
nearest_points_true = pp[nearest_inds,:]
dsqq_true = np.linalg.norm(qq - nearest_points_true, axis=1)**2

err_nearest = np.linalg.norm(nearest_points - nearest_points_true)
print('err_nearest=', err_nearest)

err_dsqq = np.linalg.norm(dsqq - dsqq_true)
print('err_dsqq=', err_dsqq)

from time import time

n_pts = int(1e5)
n_query = int(1e7)

pp = np.random.randn(n_pts, 2)
t = time()
KDT = hcpp.KDTree2D(pp)
dt_build = time() - t
print('n_pts=', n_pts, ', dt_build=', dt_build)

qq = np.random.randn(n_query, 2)
t = time()
KDT.nearest_neighbor_vectorized(qq)
dt_query = time() - t
print('n_query=', n_query, ', dt_query=', dt_query)

from scipy.spatial import KDTree

t = time()
KDT_scipy = KDTree(pp)
dt_build_scipy = time() - t
print('dt_build_scipy=', dt_build_scipy)

t = time()
KDT_scipy.query(qq)
dt_query_scipy = time() - t
print('dt_query_scipy=', dt_query_scipy)


# std::tuple<double, double>
# n_pts= 10000 , dt_build= 0.006009817123413086
# n_query= 10000000 , dt_query= 10.365345478057861
# dt_build_scipy= 0.0027518272399902344
# dt_query_scipy= 6.627758979797363

# Eigen Vector2d
# n_pts= 10000 , dt_build= 0.00654149055480957
# n_query= 10000000 , dt_query= 3.3113129138946533
# dt_build_scipy= 0.0027768611907958984
# dt_query_scipy= 6.6427671909332275

*/