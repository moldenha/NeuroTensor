#ifndef NT_LAYERS_LP_POOL_2D_H_
#define NT_LAYERS_LP_POOL_2D_H_

#include "../../Tensor.h"
#include "../Module.h"
#include "../TensorGrad.h"
#include "../../reflection/named_parameters/named_parameters.hpp"
namespace nt {
namespace layers {
class NEUROTENSOR_API LPPool2D : public Module {
  public:
    Scalar power;
    utils::my_tuple kernel_size, stride;
    bool ceil_mode;
    NT_MAKE_NAMED_ARGUMENT_CLASS_CONSTRUCTOR_(LPPool2D,
        NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_NAMES_(power, kernel_size, stride, ceil_mode),
        NT_NAMED_CLASS_CONSTRUCTOR_CLASS_DEFAULT_VALS_(-1, false),
        NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_TYPES_(Scalar, utils::my_tuple, utils::my_tuple, bool));

    TensorGrad forward(TensorGrad x);
};
} // namespace layers
} // namespace nt

#endif
