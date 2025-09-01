#ifndef NT_LAYERS_FRACTIONAL_MAX_POOL_3D_H__
#define NT_LAYERS_FRACTIONAL_MAX_POOL_3D_H__

#include "../../Tensor.h"
#include "../../utils/tuple_or_var.h"
#include "../Module.h"
#include "../TensorGrad.h"
#include "../../reflection/named_parameters/named_parameters.hpp"
#include <variant>

namespace nt {
namespace layers {
class NEUROTENSOR_API FractionalMaxPool3D : public Module {
  public:
    using ratio_type = utils::tuple_or_var<double, 3>;
    utils::my_n_tuple<3> kernel_size, output_size;
    ratio_type output_ratio;
    bool return_indices;
    NT_MAKE_NAMED_ARGUMENT_CLASS_CONSTRUCTOR_(FractionalMaxPool3D,
        NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_NAMES_(kernel_size, output_size, output_ratio, return_indices),
        NT_NAMED_CLASS_CONSTRUCTOR_CLASS_DEFAULT_VALS_(-1, double(-1.0), false),
        NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_TYPES_(utils::my_n_tuple<3>, utils::my_n_tuple<3>, ratio_type, bool));

    TensorGrad forward(TensorGrad x);
};
} // namespace layers
} // namespace nt

#endif
