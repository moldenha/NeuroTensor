#ifndef _NT_LAYERS_FOLD_H_
#define _NT_LAYERS_FOLD_H_

#include "../../Tensor.h"
#include "../Module.h"
#include "../TensorGrad.h"

namespace nt {
namespace layers {

class Fold : public Module {
  public:
    utils::my_tuple output_size, kernel_size, dilation, padding, stride;
    Fold(utils::my_tuple output_size, utils::my_tuple kernel_size,
         utils::my_tuple dilation = 1, utils::my_tuple padding = 0,
         utils::my_tuple stride = 1);
    TensorGrad forward(TensorGrad x);
};
} // namespace layers
} // namespace nt

#endif
