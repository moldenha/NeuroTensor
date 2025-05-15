#include "Unfold3D.h"
#include "../../functional/functional.h"
#include "../functional.h"
#include "../../reflection/layer_reflect/layer_registry.hpp"
#include "../../reflection/layer_reflect/reflect_macros.h"

namespace nt {
namespace layers {

Unfold3D::Unfold3D(utils::my_n_tuple<3> kernel_size,
                   utils::my_n_tuple<3> dilation, utils::my_n_tuple<3> padding,
                   utils::my_n_tuple<3> stride, bool transpose_out)
    : kernel_size(kernel_size), dilation(dilation), padding(padding),
      stride(stride), transpose_out(transpose_out) {}

TensorGrad Unfold3D::forward(TensorGrad x) {
    return functional::unfold3d(x, kernel_size, dilation, padding, stride,
                                transpose_out);
}

} // namespace layers
} // namespace nt

_NT_REGISTER_LAYER_NAMESPACED_(nt::layers::Unfold3D, nt__layers__Unfold3D,
                               kernel_size, dilation, padding, stride,
                               transpose_out)
