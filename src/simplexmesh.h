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


inline void projected_affine_coordinates( const VectorXd & query,  // shape=(dim, 1)
                                          const MatrixXd & points, // shape=(dim, npts)
                                          Ref<VectorXd>    coords) // shape=(npts, 1)
{
    int npts = points.cols();

    if ( npts == 1 )
    {
        coords(0) = 1.0;
    }
    else if ( npts == 2 )
    {
        const VectorXd dv = points.col(1) - points.col(0);
        coords(1) = dv.dot(query - points.col(0)) / dv.squaredNorm();
        coords(0) = 1.0 - coords(1);
    }
    else
    {
        const MatrixXd dV = points.rightCols(npts-1).colwise() - points.col(0);
        coords.tail(npts-1) = dV.colPivHouseholderQr().solve(query - points.col(0)); // min_c ||dV*c - b||^2
        coords(0) = 1.0 - coords.tail(npts-1).sum();
    }
}


int power_of_two( int k ) // x=2^k, with 2^0 = 1. Why does c++ not have this!?
{
    int x = 1;
    for ( int ii=1; ii<=k; ++ii )
    {
        x = 2*x;
    }
    return x;
}

inline MatrixXd select_columns( const MatrixXd & A,    // shape=(N,M)
                                const VectorXi & inds) // shape=(k,1)
{
    int N = A.rows();
    int k = inds.size();
    MatrixXd A_selected;
    A_selected.resize(N, k);
    for ( int ii=0; ii<k; ++ii)
    {
        A_selected.col(ii) = A.col(inds(ii));
    }
    return A_selected;
}

inline void closest_point_in_simplex( const VectorXd & query,            // shape=(dim, 1)
                                      const MatrixXd & simplex_vertices, // shape=(dim, npts)
                                      Ref<MatrixXd>  & closest_point )   // shape=(dim, 1)
{
    int dim = simplex_vertices.rows();
    int npts = simplex_vertices.cols();

    if ( npts == 1 )
    {
        closest_point = simplex_vertices.col(0);
    }
    else
    {
        int num_facets = power_of_two(npts) - 1;
        vector<VectorXi> all_facet_inds;
        all_facet_inds.reserve(num_facets);

        if ( npts == 2 )
        {
            VectorXi facet01_inds(1);    facet01_inds << 0;
            VectorXi facet10_inds(1);    facet10_inds << 1;
            VectorXi facet11_inds(2);    facet11_inds << 0, 1;

            all_facet_inds.push_back(facet01_inds);
            all_facet_inds.push_back(facet10_inds);
            all_facet_inds.push_back(facet11_inds);
        }
        else if ( npts == 3 )
        {
            VectorXi facet001_inds(1);    facet001_inds << 0;
            VectorXi facet010_inds(1);    facet010_inds << 1;
            VectorXi facet011_inds(2);    facet011_inds << 0, 1;
            VectorXi facet100_inds(1);    facet100_inds << 2;
            VectorXi facet101_inds(2);    facet101_inds << 0, 2;
            VectorXi facet110_inds(2);    facet110_inds << 1, 2;
            VectorXi facet111_inds(3);    facet111_inds << 0, 1, 2;

            all_facet_inds.push_back(facet001_inds);
            all_facet_inds.push_back(facet010_inds);
            all_facet_inds.push_back(facet011_inds);
            all_facet_inds.push_back(facet100_inds);
            all_facet_inds.push_back(facet101_inds);
            all_facet_inds.push_back(facet110_inds);
            all_facet_inds.push_back(facet111_inds);
        }

        closest_point = simplex_vertices.col(0);
        double dsq_best = (closest_point - query).squaredNorm();
        for ( int ii=0; ii<num_facets; ++ii ) // for each facet
        {
            MatrixXd facet_vertices = select_columns( simplex_vertices, all_facet_inds[ii] );
            VectorXd facet_coords( facet_vertices.cols() );
            projected_affine_coordinates( query, facet_vertices, facet_coords );
            bool projection_is_in_facet = (facet_coords.array() >= 0.0).all();
            if ( projection_is_in_facet )
            {
                VectorXd candidate_point = facet_vertices * facet_coords;
                double dsq_candidate = (candidate_point - query).squaredNorm();
                if ( dsq_candidate < dsq_best )
                {
                    closest_point = candidate_point;
                    dsq_best = dsq_candidate;
                }
            }
        }
//
//        closest_point = simplex_vertices.col(0);
//        double dsq_best = (closest_point - query).squaredNorm();
//        for ( int ii=0; ii<candidate_points.size(); ++ii )
//        {
//            double dsq_candidate = (candidate_points[ii] - query).squaredNorm();
//            if ( dsq_candidate < dsq_best )
//            {
//                closest_point = candidate_points[ii];
//                dsq_best = dsq_candidate;
//            }
//        }
    }

}




