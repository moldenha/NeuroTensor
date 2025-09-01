#ifndef NT_LAYERS_UNFOLD2D_H_
#define NT_LAYERS_UNFOLD2D_H_

#include "../../Tensor.h"
#include "../Module.h"
#include "../TensorGrad.h"
#include "../../reflection/named_parameters/named_parameters.hpp"
namespace nt {
namespace layers {
class NEUROTENSOR_API Unfold2D : public Module {
  public:
    utils::my_tuple kernel_size, dilation, padding, stride;
    bool transpose_out;
    NT_MAKE_NAMED_ARGUMENT_CLASS_CONSTRUCTOR_(Unfold2D,
        NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_NAMES_(kernel_size, dilation, padding, stride, transpose_out),
        NT_NAMED_CLASS_CONSTRUCTOR_CLASS_DEFAULT_VALS_(1,0,1,true),
        NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_TYPES_(utils::my_tuple, utils::my_tuple, utils::my_tuple, utils::my_tuple, bool));
    // Unfold2D(utils::my_tuple kernel_size, utils::my_tuple dilation = 1,
    //          utils::my_tuple padding = 0, utils::my_tuple stride = 1,
    //          bool transpose_out = true);
    TensorGrad forward(TensorGrad x);
};
} // namespace layers
} // namespace nt

#endif
