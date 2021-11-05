#pragma once

#include <iostream>

#include <math.h>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

pair<MatrixXd, MatrixXd> submatrix_deletion_factors( const MatrixXd &    A,
                                                     const vector<int> & bad_rows,
                                                     const vector<int> & bad_cols )
{
    MatrixXd left_factor(A.rows(), bad_rows.size() + bad_cols.size());
    left_factor.setZero();

    for ( int jj=0; jj<bad_cols.size(); ++jj )
    {
        left_factor.col(bad_rows.size() + jj) = A.col(bad_cols[jj]);
    }

    for ( int ii=0; ii<bad_rows.size(); ++ii )
    {
        left_factor.row(bad_rows[ii]).setZero();
        left_factor(bad_rows[ii], ii) = 1.0;
    }

    MatrixXd right_factor(bad_rows.size() + bad_cols.size(), A.cols());
    right_factor.setZero();

    for ( int ii=0; ii<bad_rows.size(); ++ii )
    {
        right_factor.row(ii) = -A.row(bad_rows[ii]);
    }

    for ( int jj=0; jj<bad_cols.size(); ++jj )
    {
        right_factor.col(bad_cols[jj]).setZero();
        right_factor(bad_rows.size() + jj, bad_cols[jj]) = -1.0;
    }

    return make_pair(left_factor, right_factor);
}

// Woodbury formula:
// The solution to the modified linear system:
//     (A + UV) x = b
// is given by
//     x = inv(A)*b - inv(A)*U*inv(I+V*inv(A)*U)*V*inv(A)*b
// which may be computed via
//     Step 1) x <- inv(A)*b
//     Step 2) x <- x - inv(A)*U*inv(I+V*inv(A)*U)*V*x
// This function performs step 2.
void woodbury_update( Ref<VectorXd>    x,
                      const MatrixXd & A,
//                      const MatrixXd & invA,
                      FullPivLU<MatrixXd> & factorized_A,
                      const MatrixXd & U,
                      const MatrixXd & V )
{
//    MatrixXd Z = invA * U; // Z = inv(A)*U
    MatrixXd Z = factorized_A.solve(U);
    MatrixXd C = MatrixXd::Identity(V.rows(),U.cols()) + V * Z; // C = I + V*inv(A)*U
    x -= Z * C.lu().solve(V * x); // x = x - inv(A)*U*inv(C)*V*x
}


VectorXi nearest_points_brute_force( const MatrixXd & candidate_points,
                                     const VectorXd & query_point,
                                     int k )
{
    int N = candidate_points.cols();
    vector<double> squared_distances(N);
    for ( int ii=0; ii<N; ++ii )
    {
        squared_distances[ii] = (candidate_points.col(ii) - query_point).squaredNorm();
    }
    vector<int> sort_inds(N);
    iota(sort_inds.begin(), sort_inds.end(), 0);

    sort(sort_inds.begin(), sort_inds.end(),
         [&squared_distances](int ii, int jj) {return squared_distances[ii] < squared_distances[jj];});

    VectorXi nearest_k_points(k);
    for ( int ii=0; ii<k; ++ii )
    {
        nearest_k_points(ii) = sort_inds[ii];
    }
    return nearest_k_points;
}

MatrixXi nearest_points_brute_force_vectorized( const MatrixXd & candidate_points,
                                                const MatrixXd & query_points,
                                                int k )
{
    int num_querys = query_points.cols();
    MatrixXi nearest_inds(k, num_querys);
    for ( int ii=0; ii<num_querys; ++ii )
    {
        nearest_inds.col(ii) = nearest_points_brute_force( candidate_points,
                                                           query_points.col(ii),
                                                           k );
    }
    return nearest_inds;
}