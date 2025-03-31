#ifndef _NT_LAYERS_UNFOLD3D_H_
#define _NT_LAYERS_UNFOLD3D_H_

#include "../../Tensor.h"
#include "../Module.h"
#include "../TensorGrad.h"

namespace nt {
namespace layers {

class Unfold3D : public Module {
  public:
    utils::my_n_tuple<3> kernel_size, dilation, padding, stride;
    bool transpose_out;
    Unfold3D(utils::my_n_tuple<3> kernel_size,
             utils::my_n_tuple<3> dilation = 1,
             utils::my_n_tuple<3> padding = 0, utils::my_n_tuple<3> stride = 1,
             bool transpose_out = true);
    TensorGrad forward(TensorGrad x);
};

} // namespace layers
} // namespace nt

#endif
