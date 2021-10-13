#include "mesh_operations.h"
#include <iostream>
#include <list>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
//#include <CGAL/Exact_predicates_exact_constructions_kernel.h>

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
std::vector<Point> make_CGAL_points( Array<double, Dynamic, 2> const & points_array )
{
    int num_pts = points_array.rows();
    std::vector<Point> CGAL_points;
    CGAL_points.reserve(num_pts);
    for ( int ii = 0; ii < num_pts; ++ii)
    {
        Point p(points_array(ii,0), points_array(ii,1), 0.0);
        CGAL_points.push_back(p);
    }
    return CGAL_points;
}

std::vector<Triangle> make_CGAL_triangles( Array<int, Dynamic, 3> const triangles_array,
                                           std::vector<Point> const     CGAL_points_vector )
{
    int num_triangles = triangles_array.rows();
    std::vector<Triangle> CGAL_triangles;
    CGAL_triangles.reserve(num_triangles);
    for ( int jj = 0; jj < num_triangles; ++jj )
    {
        Vector3i vertex_inds = triangles_array.row(jj);
        Point v0 = CGAL_points_vector.at(vertex_inds[0]);
        Point v1 = CGAL_points_vector.at(vertex_inds[1]);
        Point v2 = CGAL_points_vector.at(vertex_inds[2]);
        Triangle T(v0, v1, v2);
        CGAL_triangles.push_back(T);
    }
    return CGAL_triangles;
}


AABBTreeWrapper::AABBTreeWrapper( Array<double, Dynamic, 2> const & points_array,
                                  Array<int, Dynamic, 3> const &    triangles_array ) : aabb_tree{}
{
    CGAL_points = make_CGAL_points( points_array );
    TT_vector = make_CGAL_triangles(triangles_array, CGAL_points);

    std::list<Triangle> TT(TT_vector.begin(), TT_vector.end());
    aabb_tree.insert(TT.begin(), TT.end());
    aabb_tree.build();
    aabb_tree.accelerate_distance_queries();
}

Vector2d AABBTreeWrapper::closest_point( Vector2d p )
{
    Point p_CGAL(p(0), p(1), 0.0);

    Point closest_p_CGAL = aabb_tree.closest_point( p_CGAL );

    Vector2d closest_p(closest_p_CGAL[0], closest_p_CGAL[1]);
    return closest_p;
}

Array<double, Dynamic, 2> AABBTreeWrapper::closest_points( Array<double, Dynamic, 2> const & points_array )
{
    int num_points = points_array.rows();
    Array<double, Dynamic, 2> closest_points_array(num_points, 2);
    for (int ii=0; ii<num_points; ++ii)
    {
        Point p_CGAL(points_array(ii, 0), points_array(ii, 1), 0.0);
        Point closest_p_CGAL = aabb_tree.closest_point( p_CGAL );
        closest_points_array(ii, 0) = closest_p_CGAL[0];
        closest_points_array(ii, 1) = closest_p_CGAL[1];
    }
    return closest_points_array;
}



int KDTree2D::make_subtree( int start, int stop, int depth,
                            vector< Vector2d > & points,
                            int & counter ) {
    int num_pts_local = stop - start;
    int current_node_ind = -1; // -1 indicates node does not exist
    if (num_pts_local >= 1) {
        current_node_ind = counter;
        counter = counter + 1;

        int axis = depth % dim;
        sort( points.begin() + start, points.begin() + stop,
              [axis](Vector2d u, Vector2d v) {return u(axis) > v(axis);} );

        int mid = start + (num_pts_local / 2);

        int left_start = start;
        int left_stop = mid;

        int right_start = mid + 1;
        int right_stop = stop;

        int left = make_subtree(left_start, left_stop, depth + 1, points, counter);
        int right = make_subtree(right_start, right_stop, depth + 1, points, counter);

        nodes[current_node_ind] = KDTree2D::Node { points[mid], left, right }; }
    return current_node_ind; }


KDTree2D::SubtreeResult KDTree2D::nn_subtree( const Vector2d & query,
                                              int              root_index,
                                              int              depth) {
    KDTree2D::Node root = nodes[root_index];

    Vector2d delta = query - root.point;

    int best_index = root_index;
    double best_distance_squared = delta.squaredNorm();

    int axis = depth % dim;
    double displacement_to_splitting_plane = delta(axis);

    int A;
    int B;
    if (displacement_to_splitting_plane >= 0) {
        A = root.left;
        B = root.right;
    } else {
        A = root.right;
        B = root.left; }

    if (A >= 0) {
        KDTree2D::SubtreeResult nn_A = nn_subtree( query, A, depth + 1);
        if (nn_A.distance_squared < best_distance_squared) {
            best_index = nn_A.index;
            best_distance_squared = nn_A.distance_squared; } }

    if (B >= 0) {
        if (displacement_to_splitting_plane*displacement_to_splitting_plane < best_distance_squared) {
            KDTree2D::SubtreeResult nn_B = nn_subtree( query, B, depth + 1);
            if (nn_B.distance_squared < best_distance_squared) {
                best_index = nn_B.index;
                best_distance_squared = nn_B.distance_squared; } } }

    return KDTree2D::SubtreeResult { best_index, best_distance_squared }; }


KDTree2D::KDTree2D( Array<double, Dynamic, 2> & points_array ) {
    int num_pts = points_array.rows();

    // Copy eigen matrix input into std::vector of tuples which will be re-ordered
    vector< Vector2d > points(num_pts);
    for ( int ii=0; ii<num_pts; ++ii) {
        points[ii] = points_array.row(ii); }

    nodes.reserve(num_pts);
    int counter = 0;
    int zero = make_subtree(0, num_pts, 0, points, counter); }


pair<Vector2d, double> KDTree2D::nearest_neighbor( Vector2d point ) {
    KDTree2D::SubtreeResult nn_result = nn_subtree( point, 0, 0 );
    return make_pair(nodes[nn_result.index].point,
                          nn_result.distance_squared); }


pair< Array<double, Dynamic, 2>, VectorXd >
    KDTree2D::nearest_neighbor_vectorized( Array<double, Dynamic, 2> & query_array ) {
    int num_querys = query_array.rows();

    Array<double, Dynamic, 2> closest_points_array;
    closest_points_array.resize(num_querys, 2);

    VectorXd squared_distances(num_querys);

    for ( int ii=0; ii<num_querys; ++ii ) {
        Vector2d query = query_array.row(ii);
        KDTree2D::SubtreeResult nn_result = nn_subtree( query, 0, 0 );
        closest_points_array.row(ii) = nodes[nn_result.index].point;
        squared_distances(ii) = nn_result.distance_squared; }

    return make_pair(closest_points_array, squared_distances); }



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

nearest_point, dsq = KDT.nearest_neighbor(q)

nearest_ind = np.argmin(np.linalg.norm(pp - q, axis=1))
nearest_point_true = pp[nearest_ind, :]
dsq_true = np.linalg.norm(nearest_point_true - q)**2
err_nearest_one_point = np.linalg.norm(nearest_point - nearest_point_true)
err_dsq_one_point = np.abs(dsq - dsq_true)
print('err_nearest_one_point=', err_nearest_one_point)
print('err_dsq_one_point=', err_dsq_one_point)

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

n_pts = int(1e6)
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




/*
import numpy as np
import dolfin as dl
import hlibpro_python_wrapper as hpro

hcpp = hpro.hpro_cpp
mesh = dl.UnitSquareMesh(10,23)
T = hcpp.AABBTreeWrapper(mesh.coordinates(), mesh.cells())
T.closest_point(np.array([-0.3, 0.62345]))

p = np.random.randn(2)
print('p=', p)
q = T.closest_point(p)
print('q=', q)

pp = np.random.randn(100, 2) + np.array([0.5, 0.5])
pp_closest = T.closest_points(pp)
*/