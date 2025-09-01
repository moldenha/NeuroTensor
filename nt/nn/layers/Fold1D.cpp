#include "Fold1D.h"
#include "../../functional/functional.h"
#include "../functional.h"
#include "../../reflection/layer_reflect/layer_registry.hpp"
#include "../../reflection/layer_reflect/reflect_macros.h"


namespace nt {
namespace layers {

Fold1D::Fold1D(Tensor::size_value_t output_size, Tensor::size_value_t kernel_size,
       Tensor::size_value_t dilation, Tensor::size_value_t padding,
       Tensor::size_value_t stride)
      : output_size(output_size), kernel_size(kernel_size), dilation(dilation),
        padding(padding), stride(stride) {}

TensorGrad Fold1D::forward(TensorGrad x) {
    return functional::fold1d(x, output_size, kernel_size, dilation, padding,
                            stride);
}


} // namespace layers
} // namespace nt

_NT_REGISTER_LAYER_NAMESPACED_(nt::layers::Fold1D, nt__layers__Fold, output_size,
                               kernel_size, dilation, padding, stride)

