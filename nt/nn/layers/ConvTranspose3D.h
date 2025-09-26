#ifndef NT_LAYERS_CONV_TRANSPOSE3D_H__
#define NT_LAYERS_CONV_TRANSPOSE3D_H__

#include "../../Tensor.h"
#include "../Module.h"
#include "../TensorGrad.h"
#include "../../reflection/named_parameters/named_parameters.hpp"
namespace nt {
namespace layers {


class NEUROTENSOR_API ConvTranspose3D : public Module{
public:
    bool use_bias;
    int64_t groups, in_channels, out_channels;
    utils::my_n_tuple<3> stride, padding, output_padding, dilation;
    TensorGrad Weight, Bias;
    NT_MAKE_NAMED_ARGUMENT_CLASS_CONSTRUCTOR_(ConvTranspose3D,
        NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_NAMES_(in_channels, out_channels, kernel_size, stride, padding, output_padding, dilation, groups, use_bias),
        NT_NAMED_CLASS_CONSTRUCTOR_CLASS_DEFAULT_VALS_(1,0,1,1,true),
        NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_TYPES_(int64_t, int64_t, utils::my_n_tuple<3>, utils::my_n_tuple<3>, utils::my_n_tuple<3>, utils::my_n_tuple<3>, utils::my_n_tuple<3>, int64_t, bool));
    // ConvTranspose3D(int64_t in_channels, int64_t out_channels, utils::my_n_tuple<3> kernel_size,
    //        utils::my_n_tuple<3> stride = 1, utils::my_n_tuple<3> padding = 0, utils::my_n_tuple<3> output_padding = 0,
    //        utils::my_n_tuple<3> dilation = 1,
    //        int64_t groups = 1, bool use_bias = true);
    TensorGrad forward(TensorGrad x);
};

} // namespace layers
} // namespace nt

#endif
