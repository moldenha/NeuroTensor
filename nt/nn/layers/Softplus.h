#ifndef NT_LAYERS_SOFTPLUS_H_
#define NT_LAYERS_SOFTPLUS_H_

#include "../../Tensor.h"
#include "../Module.h"
#include "../TensorGrad.h"
#include "../../reflection/named_parameters/named_parameters.hpp"
namespace nt {
namespace layers {

class NEUROTENSOR_API Softplus : public Module {
  public:
    Scalar beta, threshold;
    NT_MAKE_NAMED_ARGUMENT_CLASS_CONSTRUCTOR_(Softplus,
        NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_NAMES_(beta, threshold),
        NT_NAMED_CLASS_CONSTRUCTOR_CLASS_DEFAULT_VALS_(1.0, 20.0),
        NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_TYPES_(Scalar, Scalar));
    // Softplus(Scalar beta = 1.0, Scalar threshold = 20.0);
    TensorGrad forward(TensorGrad x);
};
} // namespace layers
} // namespace nt

#endif
