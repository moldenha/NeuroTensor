#include "Unfold2D.h"
#include "../../functional/functional.h"
#include "../functional.h"
#include "../../reflection/layer_reflect/layer_registry.hpp"
#include "../../reflection/layer_reflect/reflect_macros.h"

namespace nt {
namespace layers {

Unfold2D::Unfold2D(utils::my_tuple kernel_size, utils::my_tuple dilation,
                   utils::my_tuple padding, utils::my_tuple stride,
                   bool transpose_out)
    : kernel_size(kernel_size), dilation(dilation), padding(padding),
      stride(stride), transpose_out(transpose_out) {}

TensorGrad Unfold2D::forward(TensorGrad x) {
    return functional::unfold2d(x, kernel_size, dilation, padding, stride,
                              transpose_out);
}

} // namespace layers
} // namespace nt

_NT_REGISTER_LAYER_NAMESPACED_(nt::layers::Unfold2D, nt__layers__Unfold2D,
                               kernel_size, dilation, padding, stride,
                               transpose_out)
