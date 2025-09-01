#ifndef NT_LAYERS_CONV2D_H_
#define NT_LAYERS_CONV2D_H_

#include "../../Tensor.h"
#include "../Module.h"
#include "../TensorGrad.h"
#include "../../reflection/named_parameters/named_parameters.hpp"
namespace nt {
namespace layers {
class NEUROTENSOR_API Conv2D : public Module {
  public:
    bool use_bias;
    int64_t groups, in_channels, out_channels;
    utils::my_tuple stride, padding, dilation;
    TensorGrad Weight, Bias;
    NT_MAKE_NAMED_ARGUMENT_CLASS_CONSTRUCTOR_(Conv2D,
        NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_NAMES_(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, use_bias),
        NT_NAMED_CLASS_CONSTRUCTOR_CLASS_DEFAULT_VALS_(1,0,1,1,true),
        NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_TYPES_(int64_t, int64_t, utils::my_tuple, utils::my_tuple, utils::my_tuple, utils::my_tuple, int64_t, bool));
    // Conv2D(int64_t in_channels, int64_t out_channels,
    //        utils::my_tuple kernel_size, utils::my_tuple stride = 1,
    //        utils::my_tuple padding = 0, utils::my_tuple dilation = 1,
    //        int64_t groups = 1, bool use_bias = true);
    TensorGrad forward(TensorGrad x);
};

} // namespace layers
} // namespace nt

#endif
