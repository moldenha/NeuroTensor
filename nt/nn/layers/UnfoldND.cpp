#include "UnfoldND.h"
#include "../../functional/functional.h"
#include "../functional.h"
#include "../../reflection/layer_reflect/layer_registry.hpp"
#include "../../reflection/layer_reflect/reflect_macros.h"

namespace nt {
namespace layers {

UnfoldND::UnfoldND(int64_t dim, utils::optional_list kernel_size, utils::optional_list dilation,
                   utils::optional_list padding, utils::optional_list stride,
                   bool transpose_out)
    : dim(dim), kernel_size(kernel_size), dilation(dilation), padding(padding),
      stride(stride), transpose_out(transpose_out) {}

TensorGrad UnfoldND::forward(TensorGrad x) {
    return functional::unfoldnd(x, dim, kernel_size, dilation, padding, stride,
                              transpose_out);
}

} // namespace layers
} // namespace nt

_NT_REGISTER_LAYER_NAMESPACED_(nt::layers::UnfoldND, nt__layers__UnfoldND,
                               dim, kernel_size, dilation, padding, stride,
                               transpose_out)
