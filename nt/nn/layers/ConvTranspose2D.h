#ifndef _NT_LAYERS_CONV_TRANSPOSE2D_H_
#define _NT_LAYERS_CONV_TRANSPOSE2D_H_

#include "../../Tensor.h"
#include "../Module.h"
#include "../TensorGrad.h"

namespace nt {
namespace layers {

class ConvTranspose2D : public Module {
  public:
    bool use_bias;
    int64_t groups, in_channels, out_channels;
    utils::my_tuple stride, padding, output_padding, dilation;
    TensorGrad Weight, Bias;
    ConvTranspose2D(int64_t in_channels, int64_t out_channels,
                    utils::my_tuple kernel_size, utils::my_tuple stride = 1,
                    utils::my_tuple padding = 0,
                    utils::my_tuple output_padding = 0,
                    utils::my_tuple dilation = 1, int64_t groups = 1,
                    bool use_bias = true);

    TensorGrad forward(TensorGrad x);
};

} // namespace layers
} // namespace nt

#endif
