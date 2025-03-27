#ifndef _NT_LAYERS_CONV_TRANSPOSE3D_H_
#define _NT_LAYERS_CONV_TRANSPOSE3D_H_

#include "../../Tensor.h"
#include "../Module.h"
#include "../TensorGrad.h"

namespace nt {
namespace layers {


class ConvTranspose3D : public Module{
public:
    bool use_bias;
    int64_t groups, in_channels, out_channels;
    utils::my_n_tuple<3> stride, padding, output_padding, dilation;
    TensorGrad Weight, Bias;
    ConvTranspose3D(int64_t in_channels, int64_t out_channels, utils::my_n_tuple<3> kernel_size,
           utils::my_n_tuple<3> stride = 1, utils::my_n_tuple<3> padding = 0, utils::my_n_tuple<3> output_padding = 0,
           utils::my_n_tuple<3> dilation = 1,
           int64_t groups = 1, bool use_bias = true);
    TensorGrad forward(TensorGrad x);
};

} // namespace layers
} // namespace nt

#endif
