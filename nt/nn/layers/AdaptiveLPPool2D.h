#ifndef _NT_LAYERS_ADAPTIVE_LP_POOL_2D_H_
#define _NT_LAYERS_ADAPTIVE_LP_POOL_2D_H_

#include "../../Tensor.h"
#include "../Module.h"
#include "../TensorGrad.h"
#include "../../reflection/named_parameters/named_parameters.hpp"
namespace nt {
namespace layers {
class AdaptiveLPPool2D : public Module {
  public:
    Scalar power;
    utils::my_tuple output_size;
    _NT_MAKE_NAMED_ARGUMENT_CLASS_CONSTRUCTOR_(AdaptiveLPPool2D,
        _NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_NAMES_(power, output_size),
        _NT_NAMED_CLASS_CONSTRUCTOR_CLASS_DEFAULT_VALS_(),
        _NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_TYPES_(Scalar, utils::my_tuple));

    TensorGrad forward(TensorGrad x);
};
} // namespace layers
} // namespace nt

#endif
