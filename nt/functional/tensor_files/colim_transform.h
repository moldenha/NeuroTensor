#ifndef _FUNCTIONAL_COLIM_TRANSFORM_H_
#define _FUNCTIONAL_COLIM_TRANSFORM_H_

#include "../../Tensor.h"
#include "../../utils/utils.h"


namespace nt{
namespace functional{


//2d transforms:
Tensor unfold(const Tensor&, utils::my_tuple kernel_size, utils::my_tuple dilation=1, utils::my_tuple padding = 0, utils::my_tuple stride = 1, bool transpose_out = true);
Tensor unfold_backward(const Tensor& x, const utils::my_tuple& output_size, const utils::my_tuple& kernel_size, const utils::my_tuple& dilation, const utils::my_tuple& padding, const utils::my_tuple& stride, const bool& transpose_out);
Tensor& unfold_backward(const Tensor& x, Tensor& output, const utils::my_tuple& output_size, const utils::my_tuple& kernel_size, const utils::my_tuple& dilation, const utils::my_tuple& padding, const utils::my_tuple& stride, const bool& transpose_out);



Tensor fold(const Tensor&, utils::my_tuple output_size, utils::my_tuple kernel_size, utils::my_tuple dilation=1, utils::my_tuple padding = 0, utils::my_tuple stride = 1);
Tensor fold_backward(const Tensor& grad_output, const utils::my_tuple& output_size, const utils::my_tuple& kernel_size, const utils::my_tuple& dilation, const utils::my_tuple& padding, const utils::my_tuple& stride);
Tensor& fold_backward(const Tensor& grad_output, Tensor& output, const utils::my_tuple& output_size, const utils::my_tuple& kernel_size, const utils::my_tuple& dilation, const utils::my_tuple& padding, const utils::my_tuple& stride);


//3d transforms:
Tensor unfold3d(const Tensor& x, utils::my_n_tuple<3> kernel_size, utils::my_n_tuple<3> dilation=1, utils::my_n_tuple<3> padding=0, utils::my_n_tuple<3> stride=1, bool transpose_out=true);
Tensor unfold3d_backward(const Tensor& x, const utils::my_n_tuple<3>& output_size, const utils::my_n_tuple<3>& kernel_size, const utils::my_n_tuple<3>& dilation, const utils::my_n_tuple<3>& padding, const utils::my_n_tuple<3>& stride, bool transpose_out);
Tensor& unfold3d_backward(const Tensor& x, Tensor& output, const utils::my_n_tuple<3>& output_size, const utils::my_n_tuple<3>& kernel_size, const utils::my_n_tuple<3>& dilation, const utils::my_n_tuple<3>& padding, const utils::my_n_tuple<3>& stride, bool transpose_out);


//1d transforms:
Tensor unfold1d(const Tensor& x, Tensor::size_value_t kernel_size, Tensor::size_value_t dilation=1, Tensor::size_value_t padding=0, Tensor::size_value_t stride=1, bool transpose_out=true);
Tensor unfold1d_backward(const Tensor& x, Tensor::size_value_t output_size, Tensor::size_value_t kernel_size, Tensor::size_value_t dilation, Tensor::size_value_t padding, Tensor::size_value_t stride, bool transpose_out);
Tensor& unfold1d_backward(const Tensor& x, Tensor& output, Tensor::size_value_t output_size, Tensor::size_value_t kernel_size, Tensor::size_value_t dilation, Tensor::size_value_t padding, Tensor::size_value_t stride, bool transpose_out);



}
}

#endif
