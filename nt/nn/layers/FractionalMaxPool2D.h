#ifndef NT_LAYERS_FRACTIONAL_MAX_POOL_2D_H_
#define NT_LAYERS_FRACTIONAL_MAX_POOL_2D_H_

#include "../../Tensor.h"
#include "../../utils/tuple_or_var.h"
#include "../Module.h"
#include "../TensorGrad.h"
#include "../../reflection/named_parameters/named_parameters.hpp"
#include <variant>

namespace nt {
namespace layers {
class NEUROTENSOR_API FractionalMaxPool2D : public Module {
  public:
    using ratio_type = utils::tuple_or_var<double, 2>; 
    utils::my_tuple kernel_size, output_size;
    ratio_type output_ratio;
    bool return_indices;
    NT_MAKE_NAMED_ARGUMENT_CLASS_CONSTRUCTOR_(FractionalMaxPool2D,
        NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_NAMES_(kernel_size, output_size, output_ratio, return_indices),
        NT_NAMED_CLASS_CONSTRUCTOR_CLASS_DEFAULT_VALS_(-1, double(-1.0), false),
        NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_TYPES_(utils::my_tuple, utils::my_tuple, ratio_type, bool)); //need ratio type here because there are commas in variant

    TensorGrad forward(TensorGrad x);
};
} // namespace layers
} // namespace nt

#endif
