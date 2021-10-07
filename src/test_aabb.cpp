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

std::vector<Triangle> make_CGAL_triangles( Array<int, Dynamic, 3> const & triangles_array,
                                           std::vector<Point> const &     CGAL_points_vector )
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
};

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


int main()
{
    Matrix<double, 5, 2, RowMajor> P;
    P << 0.0, 0.0,
         1.0, 0.0,
         0.0, 1.0,
         1.0, 1.0,
         1.0, 2.0;

    Matrix<int, 3, 3, RowMajor> T_inds;
    T_inds << 0, 1, 2,
              1, 2, 3,
              2, 3, 4;

//    auto P_arr = P.array();
//    Array<double, Dynamic, 2, RowMajor> P_arr = P.array();
    Array<double, Dynamic, 2> P_arr = P.array();
//    P_arr(0,0) = 99.0;
//    cout << P(0,0) << endl;
//    auto const & P_arr = P.array();

    Array<int, Dynamic, 3> triangle_inds_array = T_inds.array();

    Vector2d q_vec(0.0, 2.0);

    AABBTreeWrapper AABB_TREE(P_arr, triangle_inds_array);
    Vector2d closest_q_vec = AABB_TREE.closest_point(q_vec);
    cout << closest_q_vec(0) << ',' << closest_q_vec(1) << endl;

//    Point p0(P_arr(0,0), P_arr(0,1), 0.0);
//    Point p1(P_arr(1,0), P_arr(1,1), 0.0);
//    Point p2(P_arr(2,0), P_arr(2,1), 0.0);
//    Point p3(P_arr(3,0), P_arr(3,1), 0.0);
//    Point p4(P_arr(4,0), P_arr(4,1), 0.0);

    std::vector<Point> CGAL_points = make_CGAL_points( P_arr );

//    Point p0 = CGAL_points[0];
//    Point p1 = CGAL_points[1];
//    Point p2 = CGAL_points[2];
//    Point p3 = CGAL_points[3];
//    Point p4 = CGAL_points[4];

//    cout << p0[0] << ','<< p0[1] << endl;
//    cout << p1[0] << ','<< p1[1] << endl;
//    cout << p2[0] << ','<< p2[1] << endl;
//    cout << p3[0] << ','<< p3[1] << endl;
//    cout << p4[0] << ','<< p4[1] << endl;


//    Point p0(0.0, 0.0, 0.0);
//    Point p1(1.0, 0.0, 0.0);
//    Point p2(0.0, 1.0, 0.0);
//    Point p3(1.0, 1.0, 0.0);
//    Point p4(1.0, 2.0, 0.0);

//    Triangle T1(p0, p1, p2);
//    Triangle T2(p1, p2, p3);
//    Triangle T3(p2, p3, p4);

//    std::vector<Triangle> TT_vector;
//    TT_vector.push_back(T1);
//    TT_vector.push_back(T2);
//    TT_vector.push_back(T3);

    std::vector<Triangle> TT_vector = make_CGAL_triangles(triangle_inds_array, CGAL_points);

    std::list<Triangle> TT(TT_vector.begin(), TT_vector.end());


    Tree tree2(TT.begin(), TT.end());
//    Tree tree2(TT_vector.begin(), TT_vector.end());
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

