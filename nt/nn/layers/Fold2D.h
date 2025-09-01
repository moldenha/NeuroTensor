#ifndef NT_LAYERS_FOLD2D_H__
#define NT_LAYERS_FOLD2D_H__

#include "../../Tensor.h"
#include "../Module.h"
#include "../TensorGrad.h"
#include "../../reflection/named_parameters/named_parameters.hpp"
namespace nt {
namespace layers {

class NEUROTENSOR_API Fold2D : public Module {
  public:
    utils::my_tuple output_size, kernel_size, dilation, padding, stride;
    NT_MAKE_NAMED_ARGUMENT_CLASS_CONSTRUCTOR_(Fold2D,
        NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_NAMES_(output_size, kernel_size, dilation, padding, stride),
        NT_NAMED_CLASS_CONSTRUCTOR_CLASS_DEFAULT_VALS_(1, 0, 1),
        NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_TYPES_( utils::my_tuple, utils::my_tuple, utils::my_tuple, utils::my_tuple, utils::my_tuple));
    // Fold(utils::my_tuple output_size, utils::my_tuple kernel_size,
    //      utils::my_tuple dilation = 1, utils::my_tuple padding = 0,
    //      utils::my_tuple stride = 1);
    TensorGrad forward(TensorGrad x);
};
} // namespace layers
} // namespace nt

#endif
