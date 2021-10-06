#pragma once

#include <Eigen/Dense>
#include <Eigen/LU>

using namespace Eigen;

MatrixXd eval_thin_plate_splines_at_points(const Array<double, Dynamic, 2> & rbf_points,
                                           const Array<double, Dynamic, 2> & eval_points);

class ThinPlateSplineWeightingFunctions
{
private:
    PartialPivLU<MatrixXd> factorized_interpolation_matrix;
    const Array<double, Dynamic, 2> rbf_points;
public:
    ThinPlateSplineWeightingFunctions ( const Array<double, Dynamic, 2> rbf_points );
    MatrixXd eval_weighting_functions ( const Array<double, Dynamic, 2> & eval_points );
};
