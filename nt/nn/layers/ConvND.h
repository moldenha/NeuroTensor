#ifndef NT_LAYERS_CONVND_H__
#define NT_LAYERS_CONVND_H__

#include "../../Tensor.h"
#include "../Module.h"
#include "../TensorGrad.h"
#include "../../reflection/named_parameters/named_parameters.hpp"
namespace nt {
namespace layers {

class NEUROTENSOR_API ConvND : public Module {
  static Tensor MakeKernel(utils::optional_list, int64_t, int64_t, int64_t, int64_t);
  static Tensor MakeBias(int64_t, int64_t, bool);
  public:
    bool use_bias;
    int64_t groups, in_channels, out_channels, dim;
    utils::optional_list stride, padding, dilation;
    TensorGrad Weight, Bias;
    NT_MAKE_NAMED_ARGUMENT_CLASS_CONSTRUCTOR_(ConvND,
        NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_NAMES_(in_channels, out_channels, dim, kernel_size, stride, padding, dilation, groups, use_bias),
        NT_NAMED_CLASS_CONSTRUCTOR_CLASS_DEFAULT_VALS_(1,0,1,1,true),
        NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_TYPES_(int64_t, int64_t, int64_t, utils::optional_list, utils::optional_list, utils::optional_list, utils::optional_list, int64_t, bool));
    // ConvND(int64_t in_channels, int64_t out_channels, int64_t dim,
    //        utils::optional_list kernel_size, utils::optional_list stride = 1,
    //        utils::optional_list padding = 0, utils::optional_list dilation = 1,
    //        int64_t groups = 1, bool use_bias = true);
    TensorGrad forward(TensorGrad x);
};

} // namespace layers
} // namespace nt

#endif
