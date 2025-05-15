#include "MaxPool1D.h"
#include "../functional.h"
#include "../../functional/functional.h"
#include "../../reflection/layer_reflect/layer_registry.hpp"
#include "../../reflection/layer_reflect/reflect_macros.h"
namespace nt{
namespace layers{

MaxPool1D::MaxPool1D(int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation, bool ceil_mode, bool return_indices)
        : kernel_size(kernel_size), stride(stride), padding(padding), dilation(dilation),
          ceil_mode(ceil_mode), return_indices(return_indices)

{}

TensorGrad MaxPool1D::forward(TensorGrad x) {
    return functional::max_pool1d(x, this->kernel_size, this->stride, this->padding, this->dilation, this->ceil_mode, this->return_indices);
}

}
}


_NT_REGISTER_LAYER_NAMESPACED_(nt::layers::MaxPool1D, nt__layers__MaxPool1D, kernel_size,
                               stride, padding, dilation, ceil_mode, return_indices)

