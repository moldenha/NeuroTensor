#ifndef NT_LAYERS_CONV_TRANSPOSEND_H__
#define NT_LAYERS_CONV_TRANSPOSEND_H__

#include "../../Tensor.h"
#include "../Module.h"
#include "../TensorGrad.h"
#include "../../reflection/named_parameters/named_parameters.hpp"
namespace nt {
namespace layers {


class NEUROTENSOR_API ConvTransposeND : public Module{
public:
    bool use_bias;
    int64_t groups, dim, in_channels, out_channels;
    utils::optional_list stride, padding, output_padding, dilation;
    TensorGrad Weight, Bias;
    NT_MAKE_NAMED_ARGUMENT_CLASS_CONSTRUCTOR_(ConvTransposeND,
        NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_NAMES_(dim, in_channels, out_channels, kernel_size, stride, padding, output_padding, dilation, groups, use_bias),
        NT_NAMED_CLASS_CONSTRUCTOR_CLASS_DEFAULT_VALS_(1,0,1,1,true),
        NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_TYPES_(int64_t, int64_t, int64_t, utils::optional_list, utils::optional_list, utils::optional_list, utils::optional_list, utils::optional_list, int64_t, bool));
    // ConvTransposeND(int64_t in_channels, int64_t out_channels, utils::optional_list kernel_size,
    //        utils::optional_list stride = 1, utils::optional_list padding = 0, utils::optional_list output_padding = 0,
    //        utils::optional_list dilation = 1,
    //        int64_t groups = 1, bool use_bias = true);
    TensorGrad forward(TensorGrad x);
};

} // namespace layers
} // namespace nt

#endif
