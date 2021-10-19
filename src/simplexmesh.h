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

inline MatrixXd select_columns( const MatrixXd &                 A,           // shape=(N,M)
                                const Matrix<bool, Dynamic, 1> & is_selected) // shape=(M,1)
{
    int N = A.rows();
    int M = A.cols();
    int K = is_selected.count();
    MatrixXd A_selected;
    A_selected.resize(N, K);
    int kk = 0;
    for ( int ii=0; ii<M; ++ii)
    {
        if ( is_selected(ii) )
        {
            A_selected.col(kk) = A.col(ii);
            kk = kk + 1;
        }
    }
    return A_selected;
}

inline void closest_point_in_simplex( const VectorXd & query,            // shape=(dim, 1)
                                      const MatrixXd & simplex_vertices, // shape=(dim, npts)
                                      Ref<VectorXd>        closest_point )   // shape=(dim, 1)
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
        Matrix<bool, Dynamic, Dynamic, RowMajor> all_facet_inds;
        all_facet_inds.resize(num_facets, npts);

        if ( npts == 2 )
        {
            all_facet_inds << false, true,
                              true,  false,
                              true,  true;
        }
        else if ( npts == 3 )
        {
            all_facet_inds << false, false, true,
                              false, true,  false,
                              false, true,  true,
                              true,  false, false,
                              true,  false, true,
                              true,  true,  false,
                              true,  true,  true;
        }
        else if ( npts == 4 )
        {
            all_facet_inds << false, false, false, true,
                              false, false, true,  false,
                              false, false, true,  true,
                              false, true,  false, false,
                              false, true,  false, true,
                              false, true,  true,  false,
                              false, true,  true,  true,
                              true,  false, false, false,
                              true,  false, false, true,
                              true,  false, true,  false,
                              true,  false, true,  true,
                              true,  true,  false, false,
                              true,  true,  false, true,
                              true,  true,  true,  false,
                              true,  true,  true,  true;
        }
        else
        {
            cout << "not implemented for npts>4."
                 << "Also, algorithm not recommended for large npts since it scales combinatorially." << endl;
        }

        closest_point = simplex_vertices.col(0);
        double dsq_best = (closest_point - query).squaredNorm();
        for ( int ii=0; ii<num_facets; ++ii ) // for each facet
        {
            Matrix<bool, Dynamic, 1> facet_inds = all_facet_inds.row(ii);
            MatrixXd facet_vertices = select_columns( simplex_vertices, facet_inds );
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
    }
}

MatrixXd closest_point_in_simplex_vectorized( const MatrixXd & query,            // shape=(dim, nquery)
                                              const MatrixXd & simplex_vertices) // shape=(dim*npts, nquery)
{
    int dim = query.rows();
    int nquery = query.cols();
    int npts = simplex_vertices.rows() / dim;
    MatrixXd closest_point;
    closest_point.resize(query.rows(), query.cols());

    for ( int ii=0; ii<nquery; ++ii )
    {
        VectorXd q = query.col(ii);
        MatrixXd S;
        S.resize(dim, npts);
        for ( int jj=0; jj<dim; ++jj )
        {
            for ( int kk=0; kk<npts; ++kk )
            {
                S(jj,kk) = simplex_vertices(kk*dim + jj, ii);
            }
        }

        VectorXd p;
        p.resize(dim);

        closest_point_in_simplex( q, S, p );

        closest_point.col(ii) = p;
    }
    return closest_point;
}




