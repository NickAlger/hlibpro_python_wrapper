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
    Point p1(0.0, 0.0, 0.0);
    Point p2(1.0, 0.0, 0.0);
    Point p3(0.0, 1.0, 0.0);
    Point p4(1.0, 1.0, 0.0);
    Point p5(1.0, 2.0, 0.0);

    Triangle T1(p1, p2, p3);
    Triangle T2(p2, p3, p4);
    Triangle T3(p3, p4, p5);

    std::list<Triangle> TT;
    TT.push_back(T1);
    TT.push_back(T2);
    TT.push_back(T3);

    Tree tree2(TT.begin(), TT.end());

    Point q(0.0, 2.0, 0.0);
    Point closest_point2 = tree2.closest_point(q);
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



//// Author(s) : Pierre Alliez
//#include <iostream>
//#include <CGAL/Simple_cartesian.h>
//#include <CGAL/AABB_tree.h>
//#include <CGAL/AABB_traits.h>
//#include <CGAL/Polyhedron_3.h>
//#include <CGAL/AABB_face_graph_triangle_primitive.h>
//
//#include <CGAL/Triangle_2.h>
//#include <CGAL/Point_2.h>
//#include <list>
//
//typedef CGAL::Simple_cartesian<double> K;
//typedef K::Point_2 Point2;
//typedef K::Triangle_2 Triangle2;
//
//typedef K::FT FT;
//typedef K::Point_3 Point;
//typedef K::Segment_3 Segment;
//typedef CGAL::Polyhedron_3<K> Polyhedron;
//typedef CGAL::AABB_face_graph_triangle_primitive<Polyhedron> Primitive;
//typedef CGAL::AABB_traits<K, Primitive> Traits;
//typedef CGAL::AABB_tree<Traits> Tree;
//typedef Tree::Point_and_primitive_id Point_and_primitive_id;
//
//int main()
//{
//    Point2 u(1.5, 2.5);
//    Point2 v(-0.4, -0.2);
//    Point2 w(-0.6, 0.8);
//    Point2 z(0.0, 0.7);
//
//    Triangle2 T1(u, v, w);
//    Triangle2 T2(u, v, z);
//
//    std::list<Triangle2> TT;
//    TT.push_back(T1);
//    TT.push_back(T2);
//
//    Tree tree2(TT.begin(), TT.end());
//
//
//    Point p(1.0, 0.0, 0.0);
//    Point q(0.0, 1.0, 0.0);
//    Point r(0.0, 0.0, 1.0);
//    Point s(0.0, 0.0, 0.0);
//    Polyhedron polyhedron;
//    polyhedron.make_tetrahedron(p, q, r, s);
//
//    // constructs AABB tree and computes internal KD-tree
//    // data structure to accelerate distance queries
//    Tree tree(faces(polyhedron).first, faces(polyhedron).second, polyhedron);
//
//    // query point
//    Point query(0.0, 0.0, 3.0);
//
//    // computes squared distance from query
//    FT sqd = tree.squared_distance(query);
//    std::cout << "squared distance: " << sqd << std::endl;
//
//    // computes closest point
//    Point closest = tree.closest_point(query);
//    std::cout << "closest point: " << closest << std::endl;
//
//    // computes closest point and primitive id
//    Point_and_primitive_id pp = tree.closest_point_and_primitive(query);
//    Point closest_point = pp.first;
//    Polyhedron::Face_handle f = pp.second; // closest primitive id
//    std::cout << "closest point: " << closest_point << std::endl;
//    std::cout << "closest triangle: ( "
//              << f->halfedge()->vertex()->point() << " , "
//              << f->halfedge()->next()->vertex()->point() << " , "
//              << f->halfedge()->next()->next()->vertex()->point()
//              << " )" << std::endl;
//    return EXIT_SUCCESS;
//}



//#include <iostream>
//#include <boost/iterator/iterator_adaptor.hpp>
//#include <CGAL/Simple_cartesian.h>
//#include <CGAL/AABB_tree.h>
//#include <CGAL/AABB_traits.h>
//
//typedef CGAL::Simple_cartesian<double> K;
//
//// The points are stored in a flat array of doubles
//// The triangles are stored in a flat array of indices
//// referring to an array of coordinates: three consecutive
//// coordinates represent a point, and three consecutive
//// indices represent a triangle.
//
//typedef size_t* Point_index_iterator;
//
//// Let us now define the iterator on triangles that the tree needs:
//class Triangle_iterator
//    : public boost::iterator_adaptor<
//    Triangle_iterator               // Derived
//    , Point_index_iterator            // Base
//    , boost::use_default              // Value
//    , boost::forward_traversal_tag    // CategoryOrTraversal
//    >
//{
//public:
//    Triangle_iterator()
//        : Triangle_iterator::iterator_adaptor_() {}
//    explicit Triangle_iterator(Point_index_iterator p)
//        : Triangle_iterator::iterator_adaptor_(p) {}
//private:
//    friend class boost::iterator_core_access;
//    void increment() { this->base_reference() += 3; }
//};
//
//// The following primitive provides the conversion facilities between
//// my own triangle and point types and the CGAL ones
//struct My_triangle_primitive {
//public:
//    typedef Triangle_iterator    Id;
//    // the CGAL types returned
//    typedef K::Point_3    Point;
//    typedef K::Triangle_3 Datum;
//    // a static pointer to the vector containing the points
//    // is needed to build the triangles on the fly:
//    static const double* point_container;
//private:
//    Id m_it; // this is what the AABB tree stores internally
//public:
//    My_triangle_primitive() {} // default constructor needed
//    // the following constructor is the one that receives the iterators from the
//    // iterator range given as input to the AABB_tree
//    My_triangle_primitive(Triangle_iterator a)
//        : m_it(a) {}
//    Id id() const { return m_it; }
//    // on the fly conversion from the internal data to the CGAL types
//    Datum datum() const
//    {
//        Point_index_iterator p_it = m_it.base();
//        Point p(*(point_container + 3 * (*p_it)),
//                *(point_container + 3 * (*p_it) + 1),
//                *(point_container + 3 * (*p_it) + 2) );
//        ++p_it;
//        Point q(*(point_container + 3 * (*p_it)),
//                *(point_container + 3 * (*p_it) + 1),
//                *(point_container + 3 * (*p_it) + 2));
//        ++p_it;
//        Point r(*(point_container + 3 * (*p_it)),
//                *(point_container + 3 * (*p_it) + 1),
//                *(point_container + 3 * (*p_it) + 2));
//        return Datum(p, q, r); // assembles triangle from three points
//    }
//    // one point which must be on the primitive
//    Point reference_point() const
//    {
//      return Point(*(point_container + 3 * (*m_it)),
//                   *(point_container + 3 * (*m_it) + 1),
//                   *(point_container + 3 * (*m_it) + 2));
//    }
//};
//
//// types
//typedef CGAL::AABB_traits<K, My_triangle_primitive> My_AABB_traits;
//typedef CGAL::AABB_tree<My_AABB_traits> Tree;
//const double* My_triangle_primitive::point_container = nullptr;
//
//int main()
//{
//    // generates point set
//    double points[12];
//    My_triangle_primitive::point_container = points;
//    points[0] = 1.0; points[1] = 0.0; points[2] = 0.0;
//    points[3] = 0.0; points[4] = 1.0; points[5] = 0.0;
//    points[6] = 0.0; points[7] = 0.0; points[8] = 1.0;
//    points[9] = 0.0; points[10] = 0.0; points[11] = 0.0;
//    // generates indexed triangle set
//    size_t triangles[9];
//    triangles[0] = 0; triangles[1] = 1; triangles[2] = 2;
//    triangles[3] = 0; triangles[4] = 1; triangles[5] = 3;
//    triangles[6] = 0; triangles[7] = 3; triangles[8] = 2;
//    // constructs AABB tree
//    Tree tree(Triangle_iterator(triangles),
//        Triangle_iterator(triangles+9));
//    // counts #intersections
//    K::Ray_3 ray_query(K::Point_3(0.2, 0.2, 0.2), K::Point_3(0.0, 1.0, 0.0));
//    std::cout << tree.number_of_intersected_primitives(ray_query)
//        << " intersections(s) with ray query" << std::endl;
//    // computes closest point
//    K::Point_3 point_query(2.0, 2.0, 2.0);
//    K::Point_3 closest_point = tree.closest_point(point_query);
//    std::cout << "closest point to " << point_query << " is: " << closest_point.x() << " " << closest_point.y() << " " << closest_point.z() << std::endl;
//    return EXIT_SUCCESS;
//}