#ifndef _NT_LAYERS_ADAPTIVE_LP_POOL_3D_H_
#define _NT_LAYERS_ADAPTIVE_LP_POOL_3D_H_

#include "../../Tensor.h"
#include "../Module.h"
#include "../TensorGrad.h"
#include "../../reflection/named_parameters/named_parameters.hpp"
namespace nt {
namespace layers {
class AdaptiveLPPool3D : public Module {
  public:
    Scalar power;
    utils::my_n_tuple<3> output_size;
    _NT_MAKE_NAMED_ARGUMENT_CLASS_CONSTRUCTOR_(AdaptiveLPPool3D,
        _NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_NAMES_(power, output_size),
        _NT_NAMED_CLASS_CONSTRUCTOR_CLASS_DEFAULT_VALS_(),
        _NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_TYPES_(Scalar, utils::my_n_tuple<3>));

    TensorGrad forward(TensorGrad x);
};
} // namespace layers
} // namespace nt

#endif
