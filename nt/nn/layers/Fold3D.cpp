#include "Fold3D.h"
#include "../../functional/functional.h"
#include "../functional.h"
#include "../../reflection/layer_reflect/layer_registry.hpp"
#include "../../reflection/layer_reflect/reflect_macros.h"


namespace nt {
namespace layers {

Fold3D::Fold3D(utils::my_n_tuple<3> output_size, utils::my_n_tuple<3> kernel_size,
       utils::my_n_tuple<3> dilation, utils::my_n_tuple<3> padding,
       utils::my_n_tuple<3> stride)
      : output_size(output_size), kernel_size(kernel_size), dilation(dilation),
        padding(padding), stride(stride) {}

TensorGrad Fold3D::forward(TensorGrad x) {
    return functional::fold3d(x, output_size, kernel_size, dilation, padding,
                            stride);
}


} // namespace layers
} // namespace nt

_NT_REGISTER_LAYER_NAMESPACED_(nt::layers::Fold3D, nt__layers__Fold, output_size,
                               kernel_size, dilation, padding, stride)

