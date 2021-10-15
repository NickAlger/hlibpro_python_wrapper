#include <iostream>
#include <list>

#include <math.h>
#include <Eigen/Dense>

#include "kdtree.h"
#include "aabbtree.h"

using namespace Eigen;
using namespace std;

template <int K>
class SimplexTree
{
private:
    typedef Array<double, K, 1> KDVector;

    Array<double, dynamic, K, RowMajor> vertices;
    Array<int, dynamic, K+1, RowMajor> cells;
    Array<double, dynamic, K, RowMajor> box_mins;
    Array<double, dynamic, K, RowMajor> box_maxes;
    KDTree<K> kdtree;
    AABBTree<K> aabbtree;

public:
    SimplexTree( Array<double, dynamic, K> & input_vertices,
                 Array<double, dynamic, K> & input_cells )
    {
        // Copy input vertices into local array
        int num_vertices = input_vertices.rows();
        vertices.resize(num_vertices, K);
        for ( int ii=0; ii<num_vertices; ++ii )
        {
            vertices.row(ii) = input_vertices.row(ii);
        }

        // Copy input cells into local array
        int num_cells = input_cells.rows();
        cells.resize(num_cells, K);
        for ( int ii=0; ii<num_cells; ++ii)
        {
            cells.row(ii) = input_cells.row(ii);
        }

        // Compute box min and max points for each cell
        for ( int ii=0; ii<num_cells; ++ii)
        {
            for ( int kk=0; kk<K; ++kk )
            {
                box_mins(ii, kk) = vertices.row(cells(ii,0));
            }
        }


        kdtree = KDTree( vertices );


        aabbtree = AABBTree( box_mins, box_maxes );
    }
}