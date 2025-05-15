#ifndef _NT_LAYERS_FRACTIONAL_MAX_POOL_2D_H_
#define _NT_LAYERS_FRACTIONAL_MAX_POOL_2D_H_

#include "../../Tensor.h"
#include "../Module.h"
#include "../TensorGrad.h"
#include "../../reflection/named_parameters/named_parameters.hpp"
#include <variant>

namespace nt {
namespace layers {
class FractionalMaxPool2D : public Module {
  public:
    using ratio_type = std::variant<double, std::tuple<double, double>>; 
    utils::my_tuple kernel_size, output_size;
    ratio_type output_ratio;
    bool return_indices;
    _NT_MAKE_NAMED_ARGUMENT_CLASS_CONSTRUCTOR_(FractionalMaxPool2D,
        _NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_NAMES_(kernel_size, output_size, output_ratio, return_indices),
        _NT_NAMED_CLASS_CONSTRUCTOR_CLASS_DEFAULT_VALS_(-1, double(-1.0), false),
        _NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_TYPES_(utils::my_tuple, utils::my_tuple, ratio_type, bool)); //need ratio type here because there are commas in variant

    TensorGrad forward(TensorGrad x);
};
} // namespace layers
} // namespace nt

#endif
