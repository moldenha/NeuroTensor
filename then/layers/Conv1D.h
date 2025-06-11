#ifndef _NT_LAYERS_CONV1D_H_
#define _NT_LAYERS_CONV1D_H_

#include "../../Tensor.h"
#include "../Module.h"
#include "../TensorGrad.h"
#include "../../reflection/named_parameters/named_parameters.hpp"
namespace nt {
namespace layers {
class Conv1D : public Module {
  public:
    bool use_bias;
    int64_t groups, in_channels, out_channels;
    int64_t stride, padding, dilation;
    TensorGrad Weight, Bias;
    _NT_MAKE_NAMED_ARGUMENT_CLASS_CONSTRUCTOR_(Conv1D,
        _NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_NAMES_(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, use_bias),
        _NT_NAMED_CLASS_CONSTRUCTOR_CLASS_DEFAULT_VALS_(1,0,1,1,true),
        _NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_TYPES_(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, bool));
    // Conv1D(int64_t in_channels, int64_t out_channels, int64_t kernel_size,
    //        int64_t stride = 1, int64_t padding = 0, int64_t dilation = 1,
    //        int64_t groups = 1, bool use_bias = true);
    TensorGrad forward(TensorGrad x);
};
} // namespace layers
} // namespace nt

#endif
