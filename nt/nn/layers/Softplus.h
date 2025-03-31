#ifndef _NT_LAYERS_SOFTPLUS_H_
#define _NT_LAYERS_SOFTPLUS_H_

#include "../../Tensor.h"
#include "../Module.h"
#include "../TensorGrad.h"

namespace nt {
namespace layers {

class Softplus : public Module {
  public:
    Scalar beta, threshold;
    Softplus(Scalar beta = 1.0, Scalar threshold = 20.0);
    TensorGrad forward(TensorGrad x);
};
} // namespace layers
} // namespace nt

#endif
