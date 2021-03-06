#pragma once

#include <Eigen/Dense>
#include <Eigen/LU>
//#include <Eigen/CXX11/Tensor>

using namespace Eigen;

double grid_interpolate_at_one_point(const VectorXd p,
                                     const double xmin, const double xmax,
                                     const double ymin, const double ymax,
                                     const MatrixXd & grid_values);


VectorXd grid_interpolate(const MatrixXd & eval_coords,
                          double xmin, double xmax, double ymin, double ymax,
                          const MatrixXd & grid_values);


VectorXd grid_interpolate_vectorized(const MatrixXd & eval_coords,
                                     double xmin, double xmax, double ymin, double ymax,
                                     const MatrixXd & grid_values);

VectorXd periodic_bilinear_interpolation_regular_grid(const Array<double, Dynamic, 2> & pp,
                                                      const Vector2d & box_min,
                                                      const Vector2d & box_max,
                                                      const MatrixXd & F);

VectorXd bilinear_interpolation_regular_grid(const Array<double, Dynamic, 2> & pp,
                                             const Vector2d & box_min,
                                             const Vector2d & box_max,
                                             const MatrixXd & F);