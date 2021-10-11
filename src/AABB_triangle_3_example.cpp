// Author(s) : Camille Wormser, Pierre Alliez
#include <iostream>
#include <list>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>
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
int main()
{
    Point a(1.1, 0.0, 0.0);
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
    Point point_query0(0.5, 0.3, 0.0);
    std::cout << tree.number_of_intersected_primitives(point_query0)
        << " intersections(s) with point query" << std::endl;

//    int X = tree.any_intersected_primitive(point_query0);
//    boost::optional< typename Tree::AABB_traits::template Intersection_and_primitive_id<Triangle>::Type > X = tree.any_intersected_primitive(point_query0);
    boost::optional< Primitive::Id > X = tree.any_intersected_primitive(point_query0);

    Triangle T = *X.get();

//    Triangle T = boost::get<Triangle>(& (X->first));

    std::cout << "T=" << T << std::endl;
    std::cout << "T.vertex(0)=" << T.vertex(0) << std::endl;
    std::cout << "T.vertex(1)=" << T.vertex(1) << std::endl;
    std::cout << "T.vertex(2)=" << T.vertex(2) << std::endl;

    std::cout << "T.vertex(0)[0]=" << T.vertex(0)[0] << std::endl;

//    Triangle & Tref = *X.get(); // This works i guess?

    Point point_query0_below(point_query0[0], point_query0[1], -1.0);
    Point point_query0_above(point_query0[0], point_query0[1], 1.0);
    Line line_query0(point_query0_below, point_query0_above);

    boost::optional< Primitive::Id > Y = tree.any_intersected_primitive(line_query0);

    Triangle T2 = *Y.get();
    std::cout << "T2=" << T2 << std::endl;



    Point point_query(2.0, 2.0, 2.0);
    Point closest_point = tree.closest_point(point_query);
    std::cerr << "closest point is: " << closest_point << std::endl;
    FT sqd = tree.squared_distance(point_query);
    std::cout << "squared distance: " << sqd << std::endl;
    return EXIT_SUCCESS;
}