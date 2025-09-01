#ifndef NT_LAYERS_FOLDND_H__

#define NT_LAYERS_FOLDND_H__

#include "../../Tensor.h"
#include "../Module.h"
#include "../TensorGrad.h"
#include "../../reflection/named_parameters/named_parameters.hpp"
namespace nt {
namespace layers {

class NEUROTENSOR_API FoldND : public Module {
  public:
    int64_t dim;
    utils::optional_list output_size, kernel_size, dilation, padding, stride;
    NT_MAKE_NAMED_ARGUMENT_CLASS_CONSTRUCTOR_(FoldND,
        NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_NAMES_(dim, output_size, kernel_size, dilation, padding, stride),
        NT_NAMED_CLASS_CONSTRUCTOR_CLASS_DEFAULT_VALS_(1, 0, 1),
        NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_TYPES_( int64_t, utils::optional_list, utils::optional_list, utils::optional_list, utils::optional_list, utils::optional_list));
    // Fold(utils::my_tuple output_size, utils::my_tuple kernel_size,
    //      utils::my_tuple dilation = 1, utils::my_tuple padding = 0,
    //      utils::my_tuple stride = 1);
    TensorGrad forward(TensorGrad x);
};
} // namespace layers
} // namespace nt

#endif
