#ifndef _NT_LAYERS_CONV_TRANSPOSE1D_H_
#define _NT_LAYERS_CONV_TRANSPOSE1D_H_

#include "../../Tensor.h"
#include "../Module.h"
#include "../TensorGrad.h"

namespace nt {
namespace layers {

class ConvTranspose1D : public Module {
  public:
    bool use_bias;
    int64_t groups, in_channels, out_channels;
    int64_t stride, padding, output_padding, dilation;
    TensorGrad Weight, Bias;
    ConvTranspose1D(int64_t in_channels, int64_t out_channels,
                    int64_t kernel_size, int64_t stride = 1,
                    int64_t padding = 0, int64_t output_padding = 0,
                    int64_t dilation = 1, int64_t groups = 1,
                    bool use_bias = true);
    TensorGrad forward(TensorGrad x);
};

} // namespace layers
} // namespace nt

#endif
