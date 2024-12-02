#ifndef _FUNCTIONAL_CONV_H_
#define _FUNCTIONAL_CONV_H_
#include "functional"
#include "../utils/utils.h"
#include "../Tensor.h"

namespace nt{
namespace functional{
Tensor conv1d(const Tensor& image, const Tensor& kernel, Tensor::size_value_t stride=1, Tensor::size_value_t padding = 0, Tensor::size_value_t dilation = 1, int64_t groups=1);
Tensor conv2d(const Tensor& image, const Tensor& kernel, utils::my_tuple stride=1, utils::my_tuple padding = 0, utils::my_tuple dilation = 1, int64_t groups=1);
Tensor conv3d(const Tensor& image, const Tensor& kernel, utils::my_n_tuple<3> stride=1, utils::my_n_tuple<3> padding = 0, utils::my_n_tuple<3> dilation = 1, int64_t groups=1);


}
}


#endif
