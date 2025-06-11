#include "AvgPool2D.h"
#include "../functional.h"
#include "../../functional/functional.h"
#include "../../reflection/layer_reflect/layer_registry.hpp"
#include "../../reflection/layer_reflect/reflect_macros.h"
namespace nt{
namespace layers{

AvgPool2D::AvgPool2D(utils::my_tuple kernel_size, utils::my_tuple stride, utils::my_tuple padding, bool ceil_mode, bool count_include_pad)
        : kernel_size(kernel_size), stride(stride), padding(padding),
          ceil_mode(ceil_mode), count_include_pad(count_include_pad)

{}

TensorGrad AvgPool2D::forward(TensorGrad x) {
    return functional::avg_pool2d(x, this->kernel_size, this->stride, this->padding, this->ceil_mode, this->count_include_pad);
}

}
}


_NT_REGISTER_LAYER_NAMESPACED_(nt::layers::AvgPool2D, nt__layers__AvgPool2D, kernel_size,
                               stride, padding, ceil_mode, count_include_pad)

