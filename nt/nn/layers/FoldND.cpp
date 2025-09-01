#include "FoldND.h"
#include "../../functional/functional.h"
#include "../functional.h"
#include "../../reflection/layer_reflect/layer_registry.hpp"
#include "../../reflection/layer_reflect/reflect_macros.h"


namespace nt {
namespace layers {

FoldND::FoldND(int64_t dim, utils::optional_list output_size, utils::optional_list kernel_size,
       utils::optional_list dilation, utils::optional_list padding,
       utils::optional_list stride)
      : dim(dim), output_size(output_size), kernel_size(kernel_size), dilation(dilation),
        padding(padding), stride(stride) {}

TensorGrad FoldND::forward(TensorGrad x) {
    return functional::foldnd(x, dim, output_size, kernel_size, dilation, padding,
                            stride, false);
}


} // namespace layers
} // namespace nt

_NT_REGISTER_LAYER_NAMESPACED_(nt::layers::FoldND, nt__layers__Fold, dim, output_size,
                               kernel_size, dilation, padding, stride)

