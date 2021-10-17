#include <iostream>
#include <list>

#include <math.h>
#include <Eigen/Dense>

//#include "kdtree.h"
//#include "aabbtree.h"

using namespace Eigen;
using namespace std;

template <int K>
class SimplexMesh
{
private:
    typedef Array<double, K, 1> KDVector;

    Array<double, Dynamic, K, RowMajor> vertices;
    Array<int, Dynamic, K+1, RowMajor> cells;
    Array<double, Dynamic, K, RowMajor> box_mins;
    Array<double, Dynamic, K, RowMajor> box_maxes;
    KDTree<K> kdtree;
    AABBTree<K> aabbtree;

public:
    SimplexMesh( Array<double, Dynamic, K> & input_vertices,
                 Array<double, Dynamic, K+1> & input_cells )
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

        kdtree = KDTree<K>( vertices );
        aabbtree = AABBTree<K>( box_mins, box_maxes );
    }

};

template <int dim, int npts>
void projected_affine_coordinates( const Matrix<double, dim, 1>    & query,
                                   const Matrix<double, dim, npts> & points,
                                   Matrix<double, npts, 1>         & coords)
{
    if (dim == 1)
    {
        coords(0) = 1.0;
    }
    else
    {
        Matrix<double, dim, npts-1> X;
        for (int ii=0; ii<npts-1; ++ii)
        {
            X.col(ii) = points.col(ii+1) - points.col(0);
        }

        Matrix<double, dim, 1> b = query - points.col(0); // implicit transpose

        Matrix<double, npts, 1> coords;
        if (npts-1 == dim)
        {
            cout << "asdf" << endl;
            coords.tail(npts-1) = X.lu().solve(b);
        }
        else
        {
            Matrix<double, npts-1, dim> Xt = X.transpose();
            coords.tail(npts-1) = (Xt * X).lu().solve(Xt * b);
        }
        coords(0) = 1.0 - coords.tail(npts-1).sum();
        cout << "coords=" << coords << endl;
    }
}

template <int dim, int npts>
Matrix<double, npts, 1> projected_affine_coordinates_wrap( const Matrix<double, dim, 1>    & query,
                                                           const Matrix<double, dim, npts> & points)
{
    Matrix<double, npts, 1> coords;
    projected_affine_coordinates<dim, npts>(query, points, coords);
    return coords;
}

