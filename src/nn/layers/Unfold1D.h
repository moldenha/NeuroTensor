#ifndef _NT_LAYERS_UNFOLD1D_H_
#define _NT_LAYERS_UNFOLD1D_H_

#include "../../Tensor.h"
#include "../Module.h"
#include "../TensorGrad.h"

namespace nt {
namespace layers {
class Unfold1D : public Module {
  public:
    Tensor::size_value_t kernel_size, dilation, padding, stride;
    bool transpose_out;
    Unfold1D(Tensor::size_value_t kernel_size,
             Tensor::size_value_t dilation = 1,
             Tensor::size_value_t padding = 0, Tensor::size_value_t stride = 1,
             bool transpose_out = true);
    TensorGrad forward(TensorGrad x);
};

} // namespace layers
} // namespace nt

#endif
