#include "grid_interpolate.h"
#include <cstdlib>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/LU>

//#include <vector>

// #include <unsupported/Eigen/CXX11/Tensor>

using namespace Eigen;
using namespace std;


//Array<bool, 1, Dynamic> points_in_box_2d(const Array<double, Dynamic, 2> & pp,
//                                         const Vector2d                     box_min,
//                                         const Vector2d                     box_max)
//{
//     return (box_min(0) <= pp.col(0))  && (box_min(1) <= pp.col(1))  &&
//            (pp.col(0)  <= box_max(0)) && (pp.col(1)  <= box_max(1));
//}
//
//
//VectorXd extract_vector_entries(const VectorXd & full, const VectorXi & ind)
//{
//    int num_indices = ind.size();
//    VectorXd target(num_indices);
//    for (int i = 0; i < num_indices; i++)
//    {
//        target(i) = full(ind(i));
//    }
//    return target;
//}
//
//Array<double, 1, Dynamic> bilinear_interpolation_periodic(const Array<double, Dynamic, 2> & pp,
//                                                          const VectorXd                  & F_vectorized_rowmajor,
//                                                          const Vector2d                    box_min,
//                                                          const Vector2d                    box_max,
//                                                          const Vector2i                    grid_shape)
//{
////    int num_pts = pp.rows();
//
////    Vector2i grid_shape;
////    grid_shape << F.rows(), F.cols();
//
//    Vector2d hh;
//    hh(0) = (box_max(0) - box_min(0)) / ((double)grid_shape(0) - 1.0);
//    hh(1) = (box_max(1) - box_min(1)) / ((double)grid_shape(1) - 1.0);
//
//    Array<double, Dynamic, 2> box_coords;
//    box_coords.col(0) = (pp.col(0) - box_min(0)) / hh(0);
//    box_coords.col(1) = (pp.col(1) - box_min(1)) / hh(1);
//
//    Array<int, Dynamic, 2> lower_inds = box_coords.cast<int>();
//    Array<int, Dynamic, 2> upper_inds = lower_inds + 1;
//    Array<double, Dynamic, 2> ss = box_coords - lower_inds.cast<double>();
//
//    // mod(a, b) = a - (b * (a / b))
//    Array<int, Dynamic, 2> lower_inds_mod;
//    lower_inds_mod.col(0) = lower_inds.col(0) - (grid_shape(0) * (lower_inds.col(0) / grid_shape(0)).cast<int>());
//    lower_inds_mod.col(1) = lower_inds.col(1) - (grid_shape(1) * (lower_inds.col(1) / grid_shape(1)).cast<int>());
//
//    Array<int, Dynamic, 2> upper_inds_mod;
//    upper_inds_mod.col(0) = upper_inds.col(0) - (grid_shape(0) * (upper_inds.col(0) / grid_shape(0)).cast<int>());
//    upper_inds_mod.col(1) = upper_inds.col(1) - (grid_shape(1) * (upper_inds.col(1) / grid_shape(1)).cast<int>());
//
//    VectorXi v00_inds = grid_shape[1] * lower_inds_mod.col(0) + lower_inds_mod.col(1);
//    VectorXi v01_inds = grid_shape[1] * lower_inds_mod.col(0) + upper_inds_mod.col(1);
//    VectorXi v10_inds = grid_shape[1] * upper_inds_mod.col(0) + lower_inds_mod.col(1);
//    VectorXi v11_inds = grid_shape[1] * upper_inds_mod.col(0) + upper_inds_mod.col(1);
//
////    Array<int, 1, Dynamic> v00_inds;
////    Array<int, 1, Dynamic> v01_inds;
////    Array<int, 1, Dynamic> v10_inds;
////    Array<int, 1, Dynamic> v11_inds;
////
////    if(F.IsRowMajor)
////    {
////        v00_inds << F.cols() * lower_inds_mod.col(0) + lower_inds_mod.col(1);
////        v01_inds << F.cols() * lower_inds_mod.col(0) + upper_inds_mod.col(1);
////        v10_inds << F.cols() * upper_inds_mod.col(0) + lower_inds_mod.col(1);
////        v11_inds << F.cols() * upper_inds_mod.col(0) + upper_inds_mod.col(1);
////    }
////    else
////    {
////        v00_inds << lower_inds_mod.col(0) + F.rows() * lower_inds_mod.col(1);
////        v01_inds << lower_inds_mod.col(0) + F.rows() * upper_inds_mod.col(1);
////        v10_inds << upper_inds_mod.col(0) + F.rows() * lower_inds_mod.col(1);
////        v11_inds << upper_inds_mod.col(0) + F.rows() * upper_inds_mod.col(1);
////    }
//
////    ArrayXd v00 = F(lower_inds_mod.col(0), lower_inds_mod.col(1)); // bad
//
////    ArrayXd v00 = F(v00_inds);
////    ArrayXd v01 = F(v01_inds);
////    ArrayXd v10 = F(v10_inds);
////    ArrayXd v11 = F(v11_inds);
//
////    Map<const VectorXd> F_vectorized(F.data(), F.size());
//
////    VectorXd v00 = F_vectorized_rowmajor(v00_inds.array());
//    VectorXd v00 = extract_vector_entries(F_vectorized_rowmajor, v00_inds);
//    VectorXd v01 = extract_vector_entries(F_vectorized_rowmajor, v01_inds);
//    VectorXd v10 = extract_vector_entries(F_vectorized_rowmajor, v10_inds);
//    VectorXd v11 = extract_vector_entries(F_vectorized_rowmajor, v11_inds);
//
////    VectorXd v00 = F_vectorized(v00_inds);
////    ArrayXd v01 = F_vectorized(v01_inds);
////    ArrayXd v10 = F_vectorized(v10_inds);
////    ArrayXd v11 = F_vectorized(v11_inds);
////
//    return (1.0-ss.col(0)) * (1.0-ss.col(1)) * v00.array()
//         + (1.0-ss.col(0)) * ss.col(1)       * v01.array()
//         + ss.col(0)       * (1.0-ss.col(1)) * v10.array()
//         + ss.col(0)       * ss.col(1)       * v11.array();
//
////    Array<double, 1, Dynamic> ff;
////    return ff;
//}
//
//
////void bilinear_interpolation(Eigen::VectorXd & ff,             // function values at points, size=num_pts (to be computed: we write into this)
////                                const Eigen::MatrixXd & pp,       // points, shape=(num_pts x 2)
////                                const Eigen::MatrixXd & F,        // grid values, shape=(box_nx, box_ny)
////                                const Eigen::Vector2d & box_min,  // box minimum point, size=2
////                                const Eigen::Vector2d & box_max)  // box maximum point, size=2
////{
////    std::cout << box_min << std::endl;
//////    const int num_pts = pp.rows();
//////    const int nx = F.rows();
//////    const int ny = F.cols();
////
//////    Eigen::VectorXd in_box = (box_min(0) <= pp.col(0).array());
//////    Eigen::VectorXd in_box = (pp.array() >= -0.1);
//////    auto pp0 = pp.col(0).array();
//////    auto pp1 = pp.col(1).array();
////
//////    auto in_box = (box_min(0) <= pp0);
////
//////    auto in_box = ((box_min(0) <= pp.col(0).array()) &&
//////                   (box_min(1) <= pp.col(1).array()) &&
//////                   (pp.col(0).array() <= box_max(0)) &&
//////                   (pp.col(1).array() <= box_max(1)));
////
//////    auto in_box = ((pp.rowwise() - box_min.transpose()).array() >= 0) && ((pp.rowwise() - box_max.transpose()).array() <= 0);
////
//////    std::cout << "in_box: " << std::endl;
//////    std::cout << in_box << std::endl;
////}

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