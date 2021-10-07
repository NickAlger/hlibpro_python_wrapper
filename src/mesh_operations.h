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


class TriangulationAABBTree
{
private:
    Tree tree;
    std::vector<std::shared_ptr<Point>> CGAL_vertices;
    std::vector<std::shared_ptr<Triangle>> CGAL_triangles;
public:
    TriangulationAABBTree(const Array<double, Dynamic, 2> vertices,
                          const Array<int, Dynamic, 3>    triangles);

    Array<double, Dynamic, 2> closest_points(const Array<double, Dynamic, 2> points);
};

std::list<Triangle> make_CGAL_triangles(const Array<double, Dynamic, 2> & vertices,
                                        const Array<int, Dynamic, 3> &    triangles);

