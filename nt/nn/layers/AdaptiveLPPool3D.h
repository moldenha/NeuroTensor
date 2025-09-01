#ifndef NT_LAYERS_ADAPTIVE_LP_POOL_3D_H_
#define NT_LAYERS_ADAPTIVE_LP_POOL_3D_H_

#include "../../Tensor.h"
#include "../Module.h"
#include "../TensorGrad.h"
#include "../../reflection/named_parameters/named_parameters.hpp"
namespace nt {
namespace layers {
class NEUROTENSOR_API AdaptiveLPPool3D : public Module {
  public:
    Scalar power;
    utils::my_n_tuple<3> output_size;
    NT_MAKE_NAMED_ARGUMENT_CLASS_CONSTRUCTOR_(AdaptiveLPPool3D,
        NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_NAMES_(power, output_size),
        NT_NAMED_CLASS_CONSTRUCTOR_CLASS_DEFAULT_VALS_(),
        NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_TYPES_(Scalar, utils::my_n_tuple<3>));

    TensorGrad forward(TensorGrad x);
};
} // namespace layers
} // namespace nt

#endif
