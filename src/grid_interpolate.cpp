#include "grid_interpolate.h"
#include <cstdlib>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/LU>

//#include <vector>

// #include <unsupported/Eigen/CXX11/Tensor>

using namespace Eigen;
using namespace std;


VectorXd bilinear_interpolation_regular_grid(const Array<double, Dynamic, 2> & pp,
                                             const Vector2d & box_min,
                                             const Vector2d & box_max,
                                             const MatrixXd & F)
{
    const VectorXd ff_periodic = periodic_bilinear_interpolation_regular_grid(pp, box_min, box_max, F);

    const VectorXd box_mask = ((box_min(0) <= pp.col(0))  &&
                               (box_min(1) <= pp.col(1))  &&
                               (pp.col(0)  <= box_max(0)) &&
                               (pp.col(1)  <= box_max(1))).cast<double>();

    VectorXd ff = ff_periodic.array() * box_mask.array();
    return ff;
}

VectorXd periodic_bilinear_interpolation_regular_grid(const Array<double, Dynamic, 2> & pp,
                                                      const Vector2d & box_min,
                                                      const Vector2d & box_max,
                                                      const MatrixXd & F)
{
    const int nx = F.rows();
    const int ny = F.cols();
    const int num_pts = pp.rows();

    const double x_width = (box_max(0) - box_min(0));
    const double y_width = (box_max(1) - box_min(1));

    const int num_cells_x = nx - 1;
    const int num_cells_y = ny - 1;

    const double hx = x_width / (double)num_cells_x;
    const double hy = y_width / (double)num_cells_y;

    const VectorXd box_coords_x = (pp.col(0) - box_min(0)) / hx;
    const VectorXd box_coords_y = (pp.col(1) - box_min(1)) / hy;

    const VectorXd lower_x = box_coords_x.array().floor();
    const VectorXd lower_y = box_coords_y.array().floor();

    const VectorXd upper_x = lower_x.array() + 1.0;
    const VectorXd upper_y = lower_y.array() + 1.0;

    const VectorXd box_remainder_x = box_coords_x - lower_x;
    const VectorXd box_remainder_y = box_coords_y - lower_y;

    // mod(a, b) = a - (b * floor(a / b))
    const VectorXi lower_inds_mod_x = (lower_x.array() - (nx * (lower_x.array() / (double)nx).floor())).cast<int>();
    const VectorXi lower_inds_mod_y = (lower_y.array() - (ny * (lower_y.array() / (double)ny).floor())).cast<int>();

    const VectorXi upper_inds_mod_x = (upper_x.array() - (nx * (upper_x.array() / (double)nx).floor())).cast<int>();
    const VectorXi upper_inds_mod_y = (upper_y.array() - (ny * (upper_y.array() / (double)ny).floor())).cast<int>();

    VectorXd ff(num_pts);
    for ( int  k = 0; k < num_pts; ++k )
    {
        const double s = box_remainder_x(k);
        const double t = box_remainder_y(k);

        const int i = lower_inds_mod_x(k);
        const int i_plus = upper_inds_mod_x(k);

        const int j = lower_inds_mod_y(k);
        const int j_plus = upper_inds_mod_y(k);

        const double v00 = F(i, j);
        const double v01 = F(i,   j_plus);
        const double v10 = F(i_plus, j);
        const double v11 = F(i_plus, j_plus);

        ff(k) = (1.0-s) * (1.0-t) * v00
              + (1.0-s) * t       * v01
              + s       * (1.0-t) * v10
              + s       * t       * v11;
    }
    return ff;
}


double grid_interpolate_at_one_point(const VectorXd p,
                                     const double xmin, const double xmax,
                                     const double ymin, const double ymax,
                                     const MatrixXd & grid_values)
{
    const int nx = grid_values.rows();
    const int ny = grid_values.cols();

    double x_width = (xmax - xmin);
    double y_width = (ymax - ymin);
    double num_cells_x = nx-1;
    double num_cells_y = ny-1;
    double hx = x_width / num_cells_x;
    double hy = y_width / num_cells_y;

    double value_at_p;
    if( (p(0) < xmin) || (p(0) > xmax) || (p(1) < ymin) || (p(1) > ymax))
        value_at_p = 0.0;
    else
    {
        double quotx = (p(0) - xmin) / hx;
        int i = (int)quotx;
        double s = quotx - ((double)i);

        double quoty = (p(1) - ymin) / hy;
        int j = (int)quoty;
        double t = quoty - ((double)j);

        double v00 = grid_values(i,   j);
        double v01 = grid_values(i,   j+1);
        double v10 = grid_values(i+1, j);
        double v11 = grid_values(i+1, j+1);

        value_at_p = (1.0-s)*(1.0-t)*v00 + (1.0-s)*t*v01 + s*(1.0-t)*v10 + s*t*v11;
    }
    return value_at_p;
}

VectorXd grid_interpolate(const MatrixXd & eval_coords,
                          double xmin, double xmax, double ymin, double ymax,
                          const MatrixXd & grid_values)
{
    const int N = eval_coords.rows();
    VectorXd eval_values(N);
    eval_values.setZero();
    for ( int  k = 0; k < N; ++k )
    {
        VectorXd pk = eval_coords.row(k);
        eval_values(k) = grid_interpolate_at_one_point(pk, xmin, xmax, ymin, ymax, grid_values);
    }
    return eval_values;
}

VectorXd grid_interpolate_vectorized(const MatrixXd & eval_coords,
                                     double xmin, double xmax, double ymin, double ymax,
                                     const MatrixXd & grid_values)
{
//    int d = min_point.size()
    int d = 2;
    const int N = eval_coords.rows();
    const int nx = grid_values.rows();
    const int ny = grid_values.cols();
//    VectorXd widths = max_point - min_point
    double x_width = (xmax - xmin);
    double y_width = (ymax - ymin);
    double num_cells_x = nx-1;
    double num_cells_y = ny-1;
    double hx = x_width / num_cells_x;
    double hy = y_width / num_cells_y;

//    if(eval_coords.cols() != d)
//        throw runtime_error(std::string('points of different dimension than grid'));

    VectorXd eval_values(N);
    eval_values.setZero();
//    eval_values.resize(N);
    for ( int  k = 0; k < N; ++k )
    {
        double px = eval_coords(k,0);
        double py = eval_coords(k,1);

        if( (px < xmin) || (px >= xmax) || (py < ymin) || (py >= ymax))
            eval_values(k) = 0.0;
        else
        {
            double quotx = (px - xmin) / hx;
            int i = (int)quotx;
            double s = quotx - ((double)i);

            double quoty = (py - ymin) / hy;
            int j = (int)quoty;
            double t = quoty - ((double)j);

            double v00 = grid_values(i,   j);
            double v01 = grid_values(i,   j+1);
            double v10 = grid_values(i+1, j);
            double v11 = grid_values(i+1, j+1);

            eval_values(k) = (1.0-s)*(1.0-t)*v00 + (1.0-s)*t*v01 + s*(1.0-t)*v10 + s*t*v11;
        }
    }
    return eval_values;
}