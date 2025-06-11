#ifndef _NT_LAYERS_LP_POOL_3D_H_
#define _NT_LAYERS_LP_POOL_3D_H_

#include "../../Tensor.h"
#include "../Module.h"
#include "../TensorGrad.h"
#include "../../reflection/named_parameters/named_parameters.hpp"
namespace nt {
namespace layers {
class LPPool3D : public Module {
  public:
    Scalar power;
    utils::my_n_tuple<3> kernel_size, stride;
    bool ceil_mode;
    _NT_MAKE_NAMED_ARGUMENT_CLASS_CONSTRUCTOR_(LPPool3D,
        _NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_NAMES_(power, kernel_size, stride, ceil_mode),
        _NT_NAMED_CLASS_CONSTRUCTOR_CLASS_DEFAULT_VALS_(-1, false),
        _NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_TYPES_(Scalar, utils::my_n_tuple<3>, utils::my_n_tuple<3>, bool));

    TensorGrad forward(TensorGrad x);
};
} // namespace layers
} // namespace nt

#endif
