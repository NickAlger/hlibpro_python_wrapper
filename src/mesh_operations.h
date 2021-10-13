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


template <int K>
class KDTree {
private:
    typedef Matrix<double, K, 1> KDVector;

    // Nearest neighbor in subtree to query point
    struct SubtreeResult { int index; // index of nearest neighbor
                           double distance_squared; }; // distance squared to nearest neighbor

    VectorXi                   mids;                 // indices of splitting (mid) nodes
    VectorXi                   lefts;                // indices of children left of each node
    VectorXi                   rights;               // indices of children right of each node
    Matrix<double, K, Dynamic> ordered_points_array; // All points in the tree, depth-first ordered. column major

    // creates subtree and returns the index for root of subtree
    int make_subtree( int start, int stop, int depth,
                      vector< KDVector > & points,
                      int & counter ) {
        int num_pts_local = stop - start;
        int current = -1; // -1 indicates node does not exist
        if (num_pts_local >= 1) {
            current = counter;
            counter = counter + 1;

            int axis = depth % K;
            sort( points.begin() + start, points.begin() + stop,
                  [axis](KDVector u, KDVector v) {return u(axis) > v(axis);} );

            int mid = start + (num_pts_local / 2);
            mids(current) = mid;
            lefts(current)  = make_subtree(start,    mid, depth + 1, points, counter);
            rights(current) = make_subtree(mid + 1, stop, depth + 1, points, counter); }
        return current; }

    // finds nearest neighbor of query in subtree
    SubtreeResult nn_subtree( const KDVector & query,
                              int              current,
                              int              depth) {
        KDVector delta = query - ordered_points_array.col(mids(current));

        int best = current;
        double best_distance_squared = delta.squaredNorm();

        int axis = depth % K;
        double displacement_to_splitting_plane = delta(axis);

        int A;
        int B;
        if (displacement_to_splitting_plane >= 0) {
            A = lefts(current);
            B = rights(current);
        } else {
            A = rights(current);
            B = lefts(current); }

        if (A >= 0) {
            SubtreeResult nn_A = nn_subtree( query, A, depth + 1);
            if (nn_A.distance_squared < best_distance_squared) {
                best = nn_A.index;
                best_distance_squared = nn_A.distance_squared; } }

        if (B >= 0) {
            bool nn_might_be_in_B_subtree =
                displacement_to_splitting_plane*displacement_to_splitting_plane < best_distance_squared;
            if (nn_might_be_in_B_subtree) {
                SubtreeResult nn_B = nn_subtree( query, B, depth + 1);
                if (nn_B.distance_squared < best_distance_squared) {
                    best = nn_B.index;
                    best_distance_squared = nn_B.distance_squared; } } }

        return SubtreeResult { best, best_distance_squared }; }

public:
    KDTree( Array<double, Dynamic, K> & points_array ) {
        int num_pts = points_array.rows();

        // Copy eigen matrix input into std::vector of points which will be re-ordered
        vector< KDVector > points(num_pts);
        for ( int ii=0; ii<num_pts; ++ii) {
            points[ii] = points_array.row(ii); }

        // Make tree
        mids.resize(num_pts);
        lefts.resize(num_pts);
        rights.resize(num_pts);
        int counter = 0;
        int zero = make_subtree(0, num_pts, 0, points, counter);

        // Copy ordered vector of points into eigen matrix
        ordered_points_array.resize(K, num_pts);
        for ( int ii=0; ii<num_pts; ++ii) {
            ordered_points_array.col(ii) = points[ii]; } }

    pair<KDVector, double> nearest_neighbor( KDVector point ) {
        SubtreeResult nn_result = nn_subtree( point, 0, 0 );
        return make_pair(ordered_points_array.col(mids(nn_result.index)),
                         nn_result.distance_squared); }

    pair< Array<double, Dynamic, K>, VectorXd >
        nearest_neighbor_vectorized( Array<double, Dynamic, K> & query_array ) {
        int num_querys = query_array.rows();

        Array<double, Dynamic, K> closest_points_array;
        closest_points_array.resize(num_querys, K);

        VectorXd squared_distances(num_querys);

        for ( int ii=0; ii<num_querys; ++ii ) {
            KDVector query = query_array.row(ii);
            SubtreeResult nn_result = nn_subtree( query, 0, 0 );
            closest_points_array.row(ii) = ordered_points_array.col(mids(nn_result.index));
            squared_distances(ii) = nn_result.distance_squared; }

        return make_pair(closest_points_array, squared_distances); } };



/*
import numpy as np
import dolfin as dl
import hlibpro_python_wrapper as hpro
hcpp = hpro.hpro_cpp

K = 4

def make_KDT(pp):
    dim = pp.shape[1]
    if dim == 1:
        KDT = hcpp.KDTree1D(pp)
    elif dim == 2:
        KDT = hcpp.KDTree2D(pp)
    elif dim == 3:
        KDT = hcpp.KDTree3D(pp)
    elif dim == 4:
        KDT = hcpp.KDTree4D(pp)
    else:
        raise RuntimeError('KDT only implemented for K<=4')
    return KDT

pp = np.random.randn(100,K)
KDT = make_KDT(pp)

q = np.random.randn(K)

nearest_point, dsq = KDT.nearest_neighbor(q)

nearest_ind = np.argmin(np.linalg.norm(pp - q, axis=1))
nearest_point_true = pp[nearest_ind, :]
dsq_true = np.linalg.norm(nearest_point_true - q)**2
err_nearest_one_point = np.linalg.norm(nearest_point - nearest_point_true)
err_dsq_one_point = np.abs(dsq - dsq_true)
print('err_nearest_one_point=', err_nearest_one_point)
print('err_dsq_one_point=', err_dsq_one_point)

qq = np.random.randn(77, K)
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

pp = np.random.randn(n_pts, K)
t = time()
KDT = make_KDT(pp)
dt_build = time() - t
print('n_pts=', n_pts, ', dt_build=', dt_build)

qq = np.random.randn(n_query, K)
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

*/


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
