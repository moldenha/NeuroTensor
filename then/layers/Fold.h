#ifndef _NT_LAYERS_FOLD_H_
#define _NT_LAYERS_FOLD_H_

#include "../../Tensor.h"
#include "../Module.h"
#include "../TensorGrad.h"
#include "../../reflection/named_parameters/named_parameters.hpp"
namespace nt {
namespace layers {

class Fold : public Module {
  public:
    utils::my_tuple output_size, kernel_size, dilation, padding, stride;
    _NT_MAKE_NAMED_ARGUMENT_CLASS_CONSTRUCTOR_(Fold,
        _NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_NAMES_(output_size, kernel_size, dilation, padding, stride),
        _NT_NAMED_CLASS_CONSTRUCTOR_CLASS_DEFAULT_VALS_(1, 0, 1),
        _NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_TYPES_( utils::my_tuple, utils::my_tuple, utils::my_tuple, utils::my_tuple, utils::my_tuple));
    // Fold(utils::my_tuple output_size, utils::my_tuple kernel_size,
    //      utils::my_tuple dilation = 1, utils::my_tuple padding = 0,
    //      utils::my_tuple stride = 1);
    TensorGrad forward(TensorGrad x);
};
} // namespace layers
} // namespace nt

#endif
