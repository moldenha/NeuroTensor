#ifndef _NT_FUNCTIONAL_CONV_H_
#define _NT_FUNCTIONAL_CONV_H_
#include "../functional.h"
#include "../../utils/utils.h"
#include "../../Tensor.h"
#include "../../utils/tensor_holder.h"

namespace nt{
namespace functional{
//convolution functions
Tensor conv1d(const Tensor& image, const Tensor& kernel, Tensor::size_value_t stride=1, Tensor::size_value_t padding = 0, Tensor::size_value_t dilation = 1, int64_t groups=1, intrusive_ptr<tensor_holder> original_x = nullptr, intrusive_ptr<tensor_holder> original_w = nullptr);
Tensor conv2d(const Tensor& image, const Tensor& kernel, utils::my_tuple stride=1, utils::my_tuple padding = 0, utils::my_tuple dilation = 1, int64_t groups=1, intrusive_ptr<tensor_holder> original_x = nullptr, intrusive_ptr<tensor_holder> original_w = nullptr);
Tensor conv3d(const Tensor& image, const Tensor& kernel, utils::my_n_tuple<3> stride=1, utils::my_n_tuple<3> padding = 0, utils::my_n_tuple<3> dilation = 1, int64_t groups=1, intrusive_ptr<tensor_holder> original_x = nullptr, intrusive_ptr<tensor_holder> original_w = nullptr);

//gradient functions for the convolution
void conv_dimage(Tensor grad, Tensor kernel, Tensor& d_img, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> dilation, int64_t groups);
void conv_dkernel(Tensor grad, Tensor image, Tensor& d_kernel, std::vector<int64_t> img_size, int64_t groups);


//convolution transpose functions
Tensor conv_transpose1d(const Tensor& image, const Tensor& kernel, Tensor::size_value_t stride=1, Tensor::size_value_t padding = 0, Tensor::size_value_t output_padding = 0, Tensor::size_value_t dilation = 1, int64_t groups=1, intrusive_ptr<tensor_holder> original_x = nullptr, intrusive_ptr<tensor_holder> original_w = nullptr);
Tensor conv_transpose2d(const Tensor& image, const Tensor& kernel, utils::my_tuple stride=1, utils::my_tuple padding = 0, utils::my_tuple output_padding = 0, utils::my_tuple dilation = 1, int64_t groups=1, intrusive_ptr<tensor_holder> original_x = nullptr, intrusive_ptr<tensor_holder> original_w = nullptr);
Tensor conv_transpose3d(const Tensor& image, const Tensor& kernel, utils::my_n_tuple<3> stride=1, utils::my_n_tuple<3> padding = 0, utils::my_n_tuple<3> output_padding = 0 ,utils::my_n_tuple<3> dilation = 1, int64_t groups=1, intrusive_ptr<tensor_holder> original_x = nullptr, intrusive_ptr<tensor_holder> original_w = nullptr);

//gradient functions for convolution transpose
void convt_dimage(Tensor grad, Tensor kernel, Tensor& d_img, std::vector<int64_t> kernel_size, std::vector<int64_t> stride, std::vector<int64_t> padding, std::vector<int64_t> output_padding, std::vector<int64_t> dilation, int64_t groups);
void convt_dkernel(Tensor grad, Tensor image, Tensor& d_kernel, std::vector<int64_t> padding,  std::vector<int64_t> img_size, int64_t groups);


}
}


#endif // _NT_FUNCTIONAL_CONV_H_
