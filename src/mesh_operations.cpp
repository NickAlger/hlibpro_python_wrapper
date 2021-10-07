#include "mesh_operations.h"
#include <iostream>
#include <list>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>

#include <math.h>
#include <Eigen/Dense>

typedef CGAL::Simple_cartesian<double> K;
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
                                  Array<int, Dynamic, 3> const &    triangles_array )
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