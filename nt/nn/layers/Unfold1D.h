#ifndef NT_LAYERS_UNFOLD1D_H_
#define NT_LAYERS_UNFOLD1D_H_

#include "../../Tensor.h"
#include "../Module.h"
#include "../TensorGrad.h"
#include "../../reflection/named_parameters/named_parameters.hpp"
namespace nt {
namespace layers {
class NEUROTENSOR_API Unfold1D : public Module {
  public:
    Tensor::size_value_t kernel_size, dilation, padding, stride;
    bool transpose_out;
    NT_MAKE_NAMED_ARGUMENT_CLASS_CONSTRUCTOR_(Unfold1D,
        NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_NAMES_(kernel_size, dilation, padding, stride, transpose_out),
        NT_NAMED_CLASS_CONSTRUCTOR_CLASS_DEFAULT_VALS_(1,0,1,true),
        NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_TYPES_(Tensor::size_value_t, Tensor::size_value_t, Tensor::size_value_t, Tensor::size_value_t, bool));
    // Unfold1D(Tensor::size_value_t kernel_size,
    //          Tensor::size_value_t dilation = 1,
    //          Tensor::size_value_t padding = 0, Tensor::size_value_t stride = 1,
    //          bool transpose_out = true);
    TensorGrad forward(TensorGrad x);
};

} // namespace layers
} // namespace nt

#endif
