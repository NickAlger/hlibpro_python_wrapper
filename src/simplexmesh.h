#include <iostream>
#include <list>

#include <math.h>
#include <Eigen/Dense>

#include "kdtree.h"
#include "aabbtree.h"

using namespace Eigen;
using namespace std;

template <int K>
class SimplexTrees
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
                 Array<double, dynamic, K+1> & input_cells )
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
        for ( int cc=0; cc<num_cells; ++cc)
        {
            for ( int kk=0; kk<K; ++kk )
            {
                double min_k = vertices(cells(cc,0), kk);
                double max_k = vertices(cells(cc,0), kk);
                for ( int vv=1; vv<K+1; ++vv)
                {
                    double candidate_min_k = vertices(cells(cc,vv), kk);
                    double candidate_max_k = vertices(cells(cc,vv), kk);
                    if (candidate_min_k < min_k)
                    {
                        min_k = candidate_min_k;
                    }
                    if (candidate_max_k > max_k)
                    {
                        max_k = candidate_max_k;
                    }
                }
                box_mins(cc, kk) = min_k;
                box_maxes(cc, kk) = max_k;
            }
        }

        kdtree = KDTree( vertices );
        aabbtree = AABBTree( box_mins, box_maxes );
    }

    void project_point_onto_simplex( KDVector & point, KDVector & projected_point, int cell_id )
    {
        Matrix<double, K, K> V;
        KDVector v0 = vertices.row(cells(cell_id, 0));
        for (int vv=1; vv<K+1; ++vv)
        {
            V.row(vv-1) = vertices.row(cells(cell_id, vv)) - v0;
        }

        KDVector q = point - v0;
    }
}

