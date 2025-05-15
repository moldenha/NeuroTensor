#ifndef _NT_LAYERS_UNFOLD3D_H_
#define _NT_LAYERS_UNFOLD3D_H_

#include "../../Tensor.h"
#include "../Module.h"
#include "../TensorGrad.h"
#include "../../reflection/named_parameters/named_parameters.hpp"
namespace nt {
namespace layers {

class Unfold3D : public Module {
  public:
    utils::my_n_tuple<3> kernel_size, dilation, padding, stride;
    bool transpose_out;
    _NT_MAKE_NAMED_ARGUMENT_CLASS_CONSTRUCTOR_(Unfold3D,
        _NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_NAMES_(kernel_size, dilation, padding, stride, transpose_out),
        _NT_NAMED_CLASS_CONSTRUCTOR_CLASS_DEFAULT_VALS_(1,0,1,true),
        _NT_NAMED_CLASS_CONSTRUCTOR_CLASS_ARG_TYPES_(utils::my_n_tuple<3>, utils::my_n_tuple<3>, utils::my_n_tuple<3>, utils::my_n_tuple<3>, bool));
    // Unfold3D(utils::my_n_tuple<3> kernel_size,
    //          utils::my_n_tuple<3> dilation = 1,
    //          utils::my_n_tuple<3> padding = 0, utils::my_n_tuple<3> stride = 1,
    //          bool transpose_out = true);
    TensorGrad forward(TensorGrad x);
};

} // namespace layers
} // namespace nt

#endif
