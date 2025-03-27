#include "Unfold1D.h"
#include "../../functional/functional.h"
#include "../functional.h"
#include "../layer_reflect/layer_registry.hpp"
#include "../layer_reflect/reflect_macros.h"

namespace nt {
namespace layers {

Unfold1D::Unfold1D(Tensor::size_value_t kernel_size,
                   Tensor::size_value_t dilation, Tensor::size_value_t padding,
                   Tensor::size_value_t stride, bool transpose_out)
    : kernel_size(kernel_size), dilation(dilation), padding(padding),
      stride(stride), transpose_out(transpose_out) {}

TensorGrad Unfold1D::forward(TensorGrad x) {
    return functional::unfold1d(x, kernel_size, dilation, padding, stride,
                                transpose_out);
}

} // namespace layers
} // namespace nt

_NT_REGISTER_LAYER_NAMESPACED_(nt::layers::Unfold1D, nt__layers__Unfold1D,
                               kernel_size, dilation, padding, stride,
                               transpose_out)
