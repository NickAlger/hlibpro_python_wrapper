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

std::vector<std::shared_ptr<Point>> make_CGAL_points(const Array<double, Dynamic, 2> & points_array)
{
    int num_vertices = vertices.rows();
    std::vector<std::shared_ptr<Point>> CGAL_points;
    for ( int ii = 0; ii < num_vertices; ++ii)
    {
        std::shared_ptr<Point> p_ptr = std::make_shared<Point>(vertices(ii,0), vertices(ii,1), 0.0);
        CGAL_vertices.push_back(p_ptr);
    }
}

TriangulationAABBTree::TriangulationAABBTree(const Array<double, Dynamic, 2> vertices,
                                             const Array<int, Dynamic, 3>    triangles)
{
    int num_vertices = vertices.rows();
    int num_triangles = triangles.rows();

    for ( int ii = 0; ii < num_vertices; ++ii)
    {
        std::shared_ptr<Point> p_ptr = std::make_shared<Point>(vertices(ii,0), vertices(ii,1), 0.0);
        CGAL_vertices.push_back(p_ptr);
    }

    for ( int jj = 0; jj < num_triangles; ++jj)
    {
        const Vector3i T_verts = triangles.row(jj);
        std::shared_ptr<Triangle> T_ptr = std::make_shared<Triangle>(*CGAL_vertices[T_verts(0)],
                                                                     *CGAL_vertices[T_verts(1)],
                                                                     *CGAL_vertices[T_verts(2)]);
        CGAL_triangles.push_back(T_ptr);
    }

//    std::list<Point> CGAL_vertices_list(CGAL_vertices.begin(), CGAL_vertices.end());
//    std::list<Triangle> CGAL_triangles_list(CGAL_triangles.begin(), CGAL_triangles.end());
    std::list<Triangle> CGAL_triangles_list;
    for ( int jj = 0; jj < num_triangles; ++jj)
    {
        CGAL_triangles_list.push_back(*CGAL_triangles[jj]);
    }

    Tree tree(CGAL_triangles_list.begin(), CGAL_triangles_list.end());
    tree.accelerate_distance_queries();
}

Array<double, Dynamic, 2> TriangulationAABBTree::closest_points(const Array<double, Dynamic, 2> points)
{
    int num_points = points.rows();
    Array<double, Dynamic, 2> closest_points_array(num_points, 2);
    for ( int ii = 0; ii < num_points; ++ii)
    {
        Point p(points(ii,0), points(ii,1), 0.0);
        Point q = tree.closest_point(p);
        closest_points_array(ii,0) = q[0];
        closest_points_array(ii,1) = q[1];
    }
    return closest_points_array;
}


std::tuple< std::vector<Point>, std::vector<Triangle> > make_CGAL_points_and_triangles(const Array<double, Dynamic, 2> & vertices,
                                                                                       const Array<int, Dynamic, 3> &    triangles)
{
    int num_vertices = vertices.rows();
    int num_triangles = triangles.rows();

    std::vector<Point> CGAL_vertices;
    for ( int ii = 0; ii < num_vertices; ++ii)
    {
        Point p(vertices(ii,0), vertices(ii,1), 0.0);
        CGAL_vertices.push_back(p);
    }

    std::vector<Triangle> CGAL_triangles;
    for ( int jj = 0; jj < num_triangles; ++jj)
    {
        const Vector3i T_verts = triangles.row(jj);
        Triangle T(CGAL_vertices[T_verts(0)],
                   CGAL_vertices[T_verts(1)],
                   CGAL_vertices[T_verts(2)]);
        CGAL_triangles.push_back(T);
    }

    return std::make_tuple(CGAL_vertices, CGAL_triangles);
//
//    std::list<Triangle> CGAL_triangles_list(CGAL_triangles.begin(), CGAL_triangles.end());
//    return CGAL_triangles_list;
}

/*
import numpy as np
import dolfin as dl
import hlibpro_python_wrapper as hpro
hcpp = hpro.hpro_cpp

mesh = dl.UnitSquareMesh(10,11)
vertices = mesh.coordinates()
triangles = mesh.cells()

T = hcpp.TriangulationAABBTree(vertices, triangles)

num_pts = 20
pp = np.random.randn(num_pts, 2)

*/

//int main()
//{
//    Point p1(0.0, 0.0, 0.0);
//    Point p2(1.0, 0.0, 0.0);
//    Point p3(0.0, 1.0, 0.0);
//    Point p4(1.0, 1.0, 0.0);
//    Point p5(1.0, 2.0, 0.0);
//
//    Triangle T1(p1, p2, p3);
//    Triangle T2(p2, p3, p4);
//    Triangle T3(p3, p4, p5);
//
//    std::vector<Triangle> TT_vector;
//    TT_vector.push_back(T1);
//    TT_vector.push_back(T2);
//    TT_vector.push_back(T3);
//
//    std::list<Triangle> TT(TT_vector.begin(), TT_vector.end());
//
//
//    Tree tree2(TT.begin(), TT.end());
//    tree2.accelerate_distance_queries();
//
//    Point q(0.0, 2.0, 0.0);
//    Point closest_point2;
//    for (int ii = 0; ii < 100000; ++ii)
//    {
//        closest_point2 = tree2.closest_point(q);
//    }
//    std::cerr << "closest_point2 is: " << closest_point2 << std::endl;
//
//    FT sqd2 = tree2.squared_distance(q);
//    std::cout << "squared distance2: " << sqd2 << std::endl;
//
//
//    Point a(1.0, 0.0, 0.0);
//    Point b(0.0, 1.0, 0.0);
//    Point c(0.0, 0.0, 1.0);
//    Point d(0.0, 0.0, 0.0);
//
//    std::list<Triangle> triangles;
//    triangles.push_back(Triangle(a,b,c));
//    triangles.push_back(Triangle(a,b,d));
//    triangles.push_back(Triangle(a,d,c));
//
//    // constructs AABB tree
//    Tree tree(triangles.begin(),triangles.end());
//
//    // counts #intersections
//    Ray ray_query(a,b);
//    std::cout << tree.number_of_intersected_primitives(ray_query)
//        << " intersections(s) with ray query" << std::endl;
//
//    // compute closest point and squared distance
//    Point point_query(2.0, 2.0, 2.0);
//    Point closest_point = tree.closest_point(point_query);
//    std::cerr << "closest point is: " << closest_point << std::endl;
//    FT sqd = tree.squared_distance(point_query);
//    std::cout << "squared distance: " << sqd << std::endl;
//    return EXIT_SUCCESS;
//}
//
