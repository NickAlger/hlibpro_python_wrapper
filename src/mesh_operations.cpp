// Author(s) : Camille Wormser, Pierre Alliez
#include <iostream>
#include <list>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>

#include <math.h>
//#include <Eigen/Dense>

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

//using namespace Eigen;
using namespace std;

//std::list<Triangle> make_CGAL_triangles(const Array<double, Dynamic, 2> & vertices,
//                                        const Array<int, Dynamic, 3> &    triangles)
//{
//    int num_vertices = vertices.rows();
//    int num_triangles = triangles.rows();
//
//    std::vector<Point> CGAL_vertices;
//    for ( int ii = 0; ii < num_vertices, ++ii)
//    {
//        Point p(vertices(ii,0), vertices(ii,1), 0.0);
//        CGAL_vertices.push_back(p);
//    }
//
//    std::vector<Triangle> CGAL_triangles;
//    for ( int jj = 0; jj < num_triangles; ++jj)
//    {
//        Triangle T1(p1, p2, p3);
//    }
//}


int main()
{
    Point p1(0.0, 0.0, 0.0);
    Point p2(1.0, 0.0, 0.0);
    Point p3(0.0, 1.0, 0.0);
    Point p4(1.0, 1.0, 0.0);
    Point p5(1.0, 2.0, 0.0);

    Triangle T1(p1, p2, p3);
    Triangle T2(p2, p3, p4);
    Triangle T3(p3, p4, p5);

    std::vector<Triangle> TT_vector;
    TT_vector.push_back(T1);
    TT_vector.push_back(T2);
    TT_vector.push_back(T3);

    std::list<Triangle> TT(TT_vector.begin(), TT_vector.end());


    Tree tree2(TT.begin(), TT.end());
    tree2.accelerate_distance_queries();

    Point q(0.0, 2.0, 0.0);
    Point closest_point2;
    for (int ii = 0; ii < 100000; ++ii)
    {
        closest_point2 = tree2.closest_point(q);
    }
    std::cerr << "closest_point2 is: " << closest_point2 << std::endl;

    FT sqd2 = tree2.squared_distance(q);
    std::cout << "squared distance2: " << sqd2 << std::endl;


    Point a(1.0, 0.0, 0.0);
    Point b(0.0, 1.0, 0.0);
    Point c(0.0, 0.0, 1.0);
    Point d(0.0, 0.0, 0.0);

    std::list<Triangle> triangles;
    triangles.push_back(Triangle(a,b,c));
    triangles.push_back(Triangle(a,b,d));
    triangles.push_back(Triangle(a,d,c));

    // constructs AABB tree
    Tree tree(triangles.begin(),triangles.end());

    // counts #intersections
    Ray ray_query(a,b);
    std::cout << tree.number_of_intersected_primitives(ray_query)
        << " intersections(s) with ray query" << std::endl;

    // compute closest point and squared distance
    Point point_query(2.0, 2.0, 2.0);
    Point closest_point = tree.closest_point(point_query);
    std::cerr << "closest point is: " << closest_point << std::endl;
    FT sqd = tree.squared_distance(point_query);
    std::cout << "squared distance: " << sqd << std::endl;
    return EXIT_SUCCESS;
}

