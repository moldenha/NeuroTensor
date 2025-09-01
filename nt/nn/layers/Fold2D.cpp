#include "Fold2D.h"
#include "../../functional/functional.h"
#include "../functional.h"
#include "../../reflection/layer_reflect/layer_registry.hpp"
#include "../../reflection/layer_reflect/reflect_macros.h"


namespace nt {
namespace layers {

Fold2D::Fold2D(utils::my_tuple output_size, utils::my_tuple kernel_size,
       utils::my_tuple dilation, utils::my_tuple padding,
       utils::my_tuple stride)
      : output_size(output_size), kernel_size(kernel_size), dilation(dilation),
        padding(padding), stride(stride) {}

TensorGrad Fold2D::forward(TensorGrad x) {
    return functional::fold2d(x, output_size, kernel_size, dilation, padding,
                            stride);
}


} // namespace layers
} // namespace nt

_NT_REGISTER_LAYER_NAMESPACED_(nt::layers::Fold2D, nt__layers__Fold, output_size,
                               kernel_size, dilation, padding, stride)

