#ifndef NT_LAYERS_UNFOLDND_H_
#define NT_LAYERS_UNFOLDND_H_

#include "../../Tensor.h"
#include "../Module.h"
#include "../TensorGrad.h"
#include "../../reflection/named_parameters/named_parameters.hpp"
namespace nt {
namespace layers {
class NEUROTENSOR_API UnfoldND : public Module {
  public:
    int64_t dim;
    utils::optional_list kernel_size, dilation, padding, stride;
    bool transpose_out;
    NT_MAKE_NAMED_ARGUMENT_CLASS_CONSTRUCTOR_(UnfoldND,
        NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_NAMES_(dim, kernel_size, dilation, padding, stride, transpose_out),
        NT_NAMED_CLASS_CONSTRUCTOR_CLASS_DEFAULT_VALS_(1,0,1,true),
        NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_TYPES_(int64_t, utils::optional_list, utils::optional_list, utils::optional_list, utils::optional_list, bool));
    // UnfoldND(utils::optional_list kernel_size, utils::optional_list dilation = 1,
    //          utils::optional_list padding = 0, utils::optional_list stride = 1,
    //          bool transpose_out = true);
    TensorGrad forward(TensorGrad x);
};
} // namespace layers
} // namespace nt

#endif
