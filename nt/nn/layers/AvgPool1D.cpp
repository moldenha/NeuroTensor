#include "AvgPool1D.h"
#include "../functional.h"
#include "../../functional/functional.h"
#include "../../reflection/layer_reflect/layer_registry.hpp"
#include "../../reflection/layer_reflect/reflect_macros.h"
namespace nt{
namespace layers{

AvgPool1D::AvgPool1D(int64_t kernel_size, int64_t stride, int64_t padding, bool ceil_mode, bool count_include_pad)
        : kernel_size(kernel_size), stride(stride), padding(padding),
          ceil_mode(ceil_mode), count_include_pad(count_include_pad)

{}

TensorGrad AvgPool1D::forward(TensorGrad x) {
    return functional::avg_pool1d(x, this->kernel_size, this->stride, this->padding, this->ceil_mode, this->count_include_pad);
}

}
}


_NT_REGISTER_LAYER_NAMESPACED_(nt::layers::AvgPool1D, nt__layers__AvgPool1D, kernel_size,
                               stride, padding, ceil_mode, count_include_pad)

