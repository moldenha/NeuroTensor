#ifndef _NT_LAYERS_LINEAR_H_
#define _NT_LAYERS_LINEAR_H_

#include "../../Tensor.h"
#include "../Module.h"
#include "../TensorGrad.h"

namespace nt {
namespace layers {
class Linear : public Module {
  public:
    bool use_bias;
    TensorGrad Weight, Bias;
    Linear(int64_t in_dims, int64_t out_dims, bool use_bias = true);
    TensorGrad forward(TensorGrad);
};

} // namespace layers
} // namespace nt

#endif
