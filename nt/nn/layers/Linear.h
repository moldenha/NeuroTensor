#ifndef NT_LAYERS_LINEAR_H_
#define NT_LAYERS_LINEAR_H_

#include "../../Tensor.h"
#include "../Module.h"
#include "../TensorGrad.h"
#include "../../reflection/named_parameters/named_parameters.hpp"
namespace nt {
namespace layers {
class NEUROTENSOR_API Linear : public Module {
  public:
    bool use_bias;
    TensorGrad Weight, Bias;
    NT_MAKE_NAMED_ARGUMENT_CLASS_CONSTRUCTOR_(Linear,
        NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_NAMES_(in_channels, out_channels, use_bias),
        NT_NAMED_CLASS_CONSTRUCTOR_CLASS_DEFAULT_VALS_(true),
        NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_TYPES_(int64_t, int64_t, bool));
    // Linear(int64_t in_channels, int64_t out_channels, bool use_bias = true);
    TensorGrad forward(TensorGrad);
};

} // namespace layers
} // namespace nt

#endif
