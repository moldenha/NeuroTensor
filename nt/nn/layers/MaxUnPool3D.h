#ifndef _NT_LAYERS_MAX_UN_POOL_3D_H_
#define _NT_LAYERS_MAX_UN_POOL_3D_H_

#include "../../Tensor.h"
#include "../Module.h"
#include "../TensorGrad.h"
#include "../../reflection/named_parameters/named_parameters.hpp"
namespace nt {
namespace layers {
class MaxUnPool3D : public Module {
  public:
    utils::my_n_tuple<3> kernel_size, stride, padding, output_size;
    _NT_MAKE_NAMED_ARGUMENT_CLASS_CONSTRUCTOR_(MaxUnPool3D,
        _NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_NAMES_(kernel_size, stride, padding, output_size),
        _NT_NAMED_CLASS_CONSTRUCTOR_CLASS_DEFAULT_VALS_(-1, 0, -1),
        _NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_TYPES_(utils::my_n_tuple<3>, utils::my_n_tuple<3>, utils::my_n_tuple<3>, utils::my_n_tuple<3>));

    TensorGrad forward(TensorGrad x, TensorGrad indices);
};
} // namespace layers
} // namespace nt

#endif
