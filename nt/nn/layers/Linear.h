#ifndef _NT_LAYERS_LINEAR_H_
#define _NT_LAYERS_LINEAR_H_

#include "../../Tensor.h"
#include "../Module.h"
#include "../TensorGrad.h"
#include "../../reflection/named_parameters/named_parameters.hpp"
namespace nt {
namespace layers {
class Linear : public Module {
  public:
    bool use_bias;
    TensorGrad Weight, Bias;
    _NT_MAKE_NAMED_ARGUMENT_CLASS_CONSTRUCTOR_(Linear,
        _NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_NAMES_(in_channels, out_channels, use_bias),
        _NT_NAMED_CLASS_CONSTRUCTOR_CLASS_DEFAULT_VALS_(true),
        _NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_TYPES_(int64_t, int64_t, bool));
    // Linear(int64_t in_channels, int64_t out_channels, bool use_bias = true);
    TensorGrad forward(TensorGrad);
};

} // namespace layers
} // namespace nt

#endif
