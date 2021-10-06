#include "rbf_interpolation.h"
#include <cstdlib>
#include <iostream>
#include <math.h>
#include <Eigen/Dense>
#include <Eigen/LU>


using namespace Eigen;
using namespace std;

MatrixXd eval_thin_plate_splines_at_points(const Array<double, Dynamic, 2> & rbf_points,
                                           const Array<double, Dynamic, 2> & eval_points)
{
    const int num_rbf_points = rbf_points.rows();
    const int num_eval_points = eval_points.rows();
    MatrixXd ff(num_rbf_points, num_eval_points);
    for ( int  ii = 0; ii < num_rbf_points; ++ii )
    {
        for ( int jj = 0; jj < num_eval_points; ++jj )
        {
            const Vector2d delta = eval_points.row(jj) - rbf_points.row(ii);
            const double r_squared = delta.squaredNorm();
            if( r_squared == 0.0 )
                ff(ii, jj) = 0.0;
            else
            {
                const double r = sqrt(r_squared);
                ff(ii, jj) = r_squared * log(r);
            }
        }
    }
    return ff;
}

ThinPlateSplineWeightingFunctions::ThinPlateSplineWeightingFunctions( const Array<double, Dynamic, 2> rbf_points ) :
    rbf_points(rbf_points)
{
    MatrixXd interpolation_matrix = eval_thin_plate_splines_at_points(rbf_points, rbf_points);
    factorized_interpolation_matrix.compute(interpolation_matrix);
}

MatrixXd ThinPlateSplineWeightingFunctions::eval_weighting_functions( const Array<double, Dynamic, 2> & eval_points )
{
    MatrixXd ff = eval_thin_plate_splines_at_points( rbf_points, eval_points );
    return factorized_interpolation_matrix.solve(ff);
}


