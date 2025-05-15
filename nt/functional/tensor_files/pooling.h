#ifndef __NT_FUNCTIONAL_TENSOR_FILES_POOLING_H__
#define __NT_FUNCTIONAL_TENSOR_FILES_POOLING_H__

#include "../../Tensor.h"
#include "../../dtype/Scalar.h"
#include "../../utils/utils.h"
#include <vector>

namespace nt{
namespace functional{

//Average Pooling
Tensor avg_pool1d(Tensor input, int64_t kernel_size, int64_t stride = -1, int64_t padding = 0, bool ceil_mode = false, bool count_include_pad = true);
Tensor backward_avg_pool1d(SizeRef in_shape, Tensor output_grad, int64_t kernel_size, int64_t stride = -1, int64_t padding = 0, bool ceil_mode = false, bool count_include_pad = true);
Tensor adaptive_avg_pool1d(Tensor x, int64_t l_out);

Tensor avg_pool2d(Tensor input, utils::my_tuple kernel_size, utils::my_tuple stride = -1, utils::my_tuple padding = 0, 
                      bool ceil_mode = false, bool count_include_pad = true);
Tensor backward_avg_pool2d(SizeRef in_shape, Tensor output_grad, 
                               utils::my_tuple kernel_size, utils::my_tuple stride = -1, utils::my_tuple padding = 0, 
                               bool ceil_mode = false, bool count_include_pad = true);
Tensor adaptive_avg_pool2d(Tensor x, utils::my_tuple out_shape);

Tensor avg_pool3d(Tensor input, utils::my_n_tuple<3> kernel_size, utils::my_n_tuple<3> stride = -1, utils::my_n_tuple<3> padding = 0, 
                      bool ceil_mode = false, bool count_include_pad = true);
Tensor backward_avg_pool3d(SizeRef in_shape, Tensor output_grad, 
                               utils::my_n_tuple<3> kernel_size, utils::my_n_tuple<3> stride = -1, utils::my_n_tuple<3> padding = 0, 
                               bool ceil_mode = false, bool count_include_pad = true);
Tensor adaptive_avg_pool3d(Tensor x, utils::my_n_tuple<3> out_shape);

//LP Pooling
Tensor lp_pool1d(Tensor input, Scalar power, int64_t kernel_size, int64_t stride = -1, bool ceil_mode = false);
Tensor backward_lp_pool1d(Tensor input, Tensor output_grad, 
                                Scalar power, int64_t kernel_size, int64_t stride = -1,
                                bool ceil_mode = false);
Tensor adaptive_lp_pool1d(Tensor x, int64_t l_out, Scalar power);

Tensor lp_pool2d(Tensor input, Scalar power, utils::my_tuple kernel_size, utils::my_tuple stride = -1, bool ceil_mode = false);
Tensor backward_lp_pool2d(Tensor input, Tensor output_grad, 
                                Scalar power, utils::my_tuple kernel_size, utils::my_tuple stride = -1,
                                bool ceil_mode = false);
Tensor adaptive_lp_pool2d(Tensor x, utils::my_tuple out_shape, Scalar power);

Tensor lp_pool3d(Tensor input, Scalar power, utils::my_n_tuple<3> kernel_size, utils::my_n_tuple<3> stride = -1, bool ceil_mode = false);
Tensor backward_lp_pool3d(Tensor input, Tensor output_grad, 
                                Scalar power, utils::my_n_tuple<3> kernel_size, utils::my_n_tuple<3> stride = -1,
                                bool ceil_mode = false);
Tensor adaptive_lp_pool3d(Tensor x, utils::my_n_tuple<3> out_shape, Scalar power);


//Max Pooling
Tensor max_pool1d(Tensor input, int64_t kernel_size, int64_t stride = -1, int64_t padding = 0, int64_t dilation=1, bool ceil_mode = false, bool return_indices = false, bool get_bools = false);
Tensor backward_max_pool1d(SizeRef input_shape, Tensor dldg, Tensor indices);
Tensor max_unpool1d(Tensor input, Tensor indices, int64_t kernel_size, int64_t stride = -1, int64_t padding = 0, int64_t output_size=-1);
Tensor backward_max_unpool1d(SizeRef input_shape, Tensor dldg, Tensor indices, int64_t padding);
Tensor adaptive_max_pool1d(Tensor x, int64_t l_out, bool return_indices = false);

Tensor max_pool2d(Tensor input, 
                      utils::my_tuple kernel_size, utils::my_tuple stride = -1, utils::my_tuple padding = 0,
                      utils::my_tuple dilation=1, bool ceil_mode = false, bool return_indices = false, bool get_bools = false);
Tensor backward_max_pool2d(SizeRef input_shape, Tensor dldg, Tensor indices);
Tensor max_unpool2d(Tensor input, Tensor indices, utils::my_tuple kernel_size, utils::my_tuple stride, utils::my_tuple padding, utils::my_tuple output_size = -1);
Tensor backward_max_unpool2d(SizeRef input_shape, Tensor dldg, Tensor indices, utils::my_tuple padding);
Tensor adaptive_max_pool2d(Tensor x, utils::my_tuple out_shape, bool return_indices = false);

Tensor max_pool3d(Tensor input, 
                      utils::my_n_tuple<3> kernel_size, utils::my_n_tuple<3> stride = -1, utils::my_n_tuple<3> padding = 0,
                      utils::my_n_tuple<3> dilation=1, bool ceil_mode = false, bool return_indices = false, bool get_bools = false);
Tensor backward_max_pool3d(SizeRef input_shape, Tensor dldg, Tensor indices);
Tensor max_unpool3d(Tensor input, Tensor indices, utils::my_n_tuple<3> kernel_size, utils::my_n_tuple<3> stride, utils::my_n_tuple<3> padding, utils::my_n_tuple<3> output_size = -1);
Tensor backward_max_unpool3d(SizeRef input_shape, Tensor dldg, Tensor indices,
                             utils::my_n_tuple<3> padding);
Tensor adaptive_max_pool3d(Tensor x, utils::my_n_tuple<3> out_shape, bool return_indices = false);


//Fractional Max Pooling
Tensor extract_sliding_windows_max_2d(const std::vector<int64_t>& rows, const std::vector<int64_t>& cols, Tensor input);
Tensor fractional_max_pool2d(Tensor input, utils::my_tuple kernel_size, utils::my_tuple output_size = -1, 
                                 std::variant<double, std::tuple<double, double>> output_ratio = double(-1.0), bool return_indices = false);
Tensor extract_sliding_windows_max_3d(const std::vector<int64_t>& channels, const std::vector<int64_t>& rows, const std::vector<int64_t>& cols, Tensor input);
Tensor fractional_max_pool3d(Tensor input, utils::my_n_tuple<3> kernel_size, utils::my_n_tuple<3> output_size = -1, 
                                 std::variant<double, std::tuple<double, double, double>> output_ratio = double(-1.0), bool return_indices = false);

}
}

#endif
