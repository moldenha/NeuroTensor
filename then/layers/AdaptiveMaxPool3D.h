#ifndef _NT_LAYERS_ADAPTIVE_MAX_POOL_3D_H_
#define _NT_LAYERS_ADAPTIVE_MAX_POOL_3D_H_

#include "../../Tensor.h"
#include "../Module.h"
#include "../TensorGrad.h"
#include "../../reflection/named_parameters/named_parameters.hpp"
namespace nt {
namespace layers {
class AdaptiveMaxPool3D : public Module {
  public:
    utils::my_n_tuple<3> output_size;
    bool return_indices;
    _NT_MAKE_NAMED_ARGUMENT_CLASS_CONSTRUCTOR_(AdaptiveMaxPool3D,
        _NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_NAMES_(output_size, return_indices),
        _NT_NAMED_CLASS_CONSTRUCTOR_CLASS_DEFAULT_VALS_(false),
        _NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_TYPES_(utils::my_n_tuple<3>, bool));

    TensorGrad forward(TensorGrad x);
};
} // namespace layers
} // namespace nt

#endif
