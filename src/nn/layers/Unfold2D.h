#ifndef _NT_LAYERS_UNFOLD2D_H_
#define _NT_LAYERS_UNFOLD2D_H_

#include "../../Tensor.h"
#include "../Module.h"
#include "../TensorGrad.h"

namespace nt {
namespace layers {
class Unfold2D : public Module {
  public:
    utils::my_tuple kernel_size, dilation, padding, stride;
    bool transpose_out;
    Unfold2D(utils::my_tuple kernel_size, utils::my_tuple dilation = 1,
             utils::my_tuple padding = 0, utils::my_tuple stride = 1,
             bool transpose_out = true);
    TensorGrad forward(TensorGrad x);
};
} // namespace layers
} // namespace nt

#endif
