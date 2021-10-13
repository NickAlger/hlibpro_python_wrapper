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


class KDTree2D {
private:
    // Node in KD tree
    struct Node { Vector2d point;
                  int      left;       // index of left child
                  int      right; };   // index of right child

    // Nearest neighbor in subtree to query point
    struct SubtreeResult { int index; // index of nearest neighbor
                           double distance_squared; }; // distance squared to nearest neighbor

    vector< Node > nodes; // All nodes in the tree
    int dim = 2; // spatial dimension

    // creates subtree and returns the index for root of subtree
    int make_subtree( int start, int stop, int depth,
                     vector< Vector2d > & points,
                     int & counter );

    // finds nearest neighbor of query in subtree
    SubtreeResult nn_subtree( const Vector2d & query,
                              int              root_index, // index for the root of the subtree
                              int              depth);

public:
    KDTree2D( Array<double, Dynamic, 2> & points_array );

    pair<Vector2d, double> nearest_neighbor( Vector2d point );

    pair< Array<double, Dynamic, 2>, VectorXd >
        nearest_neighbor_vectorized( Array<double, Dynamic, 2> & query_array ); };
