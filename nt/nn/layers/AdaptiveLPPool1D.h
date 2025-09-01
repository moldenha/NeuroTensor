#ifndef NT_LAYERS_ADAPTIVE_LP_POOL_1D_H_
#define NT_LAYERS_ADAPTIVE_LP_POOL_1D_H_

#include "../../Tensor.h"
#include "../Module.h"
#include "../TensorGrad.h"
#include "../../reflection/named_parameters/named_parameters.hpp"
namespace nt {
namespace layers {
class NEUROTENSOR_API AdaptiveLPPool1D : public Module {
  public:
    Scalar power;
    int64_t output_size;
    NT_MAKE_NAMED_ARGUMENT_CLASS_CONSTRUCTOR_(AdaptiveLPPool1D,
        NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_NAMES_(power, output_size),
        NT_NAMED_CLASS_CONSTRUCTOR_CLASS_DEFAULT_VALS_(),
        NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_TYPES_(Scalar, int64_t));

    TensorGrad forward(TensorGrad x);
};
} // namespace layers
} // namespace nt

#endif
