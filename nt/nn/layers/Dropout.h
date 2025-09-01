#ifndef NT_LAYERS_DROPOUT_H_
#define NT_LAYERS_DROPOUT_H_

#include "../../Tensor.h"
#include "../Module.h"
#include "../TensorGrad.h"

namespace nt {
namespace layers {

class NEUROTENSOR_API Dropout : public Module {
  public:
    double p;
    Dropout(Scalar s);
    Dropout(double s);
    TensorGrad forward(TensorGrad x);
};

} // namespace layers
} // namespace nt

#endif
