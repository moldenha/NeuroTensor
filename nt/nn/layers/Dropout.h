#ifndef _NT_LAYERS_DROPOUT_H_
#define _NT_LAYERS_DROPOUT_H_

#include "../../Tensor.h"
#include "../Module.h"
#include "../TensorGrad.h"

namespace nt {
namespace layers {

class Dropout : public Module {
  public:
    double p;
    Dropout(Scalar s);
    Dropout(double s);
    TensorGrad forward(TensorGrad x);
};

} // namespace layers
} // namespace nt

#endif
