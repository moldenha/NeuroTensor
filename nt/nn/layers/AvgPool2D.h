#ifndef NT_LAYERS_AVG_POOL_2D_H_
#define NT_LAYERS_AVG_POOL_2D_H_

#include "../../Tensor.h"
#include "../Module.h"
#include "../TensorGrad.h"
#include "../../reflection/named_parameters/named_parameters.hpp"
namespace nt {
namespace layers {
class NEUROTENSOR_API AvgPool2D : public Module {
  public:
    utils::my_tuple kernel_size, stride, padding;
    bool ceil_mode, count_include_pad;
    NT_MAKE_NAMED_ARGUMENT_CLASS_CONSTRUCTOR_(AvgPool2D,
        NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_NAMES_(kernel_size, stride, padding, ceil_mode, count_include_pad),
        NT_NAMED_CLASS_CONSTRUCTOR_CLASS_DEFAULT_VALS_(-1, 0, false, true),
        NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_TYPES_(utils::my_tuple, utils::my_tuple, utils::my_tuple, bool, bool));

    TensorGrad forward(TensorGrad x);
};
} // namespace layers
} // namespace nt

#endif
