#include "MaxPool2D.h"
#include "../functional.h"
#include "../../functional/functional.h"
#include "../../reflection/layer_reflect/layer_registry.hpp"
#include "../../reflection/layer_reflect/reflect_macros.h"
namespace nt{
namespace layers{

MaxPool2D::MaxPool2D(utils::my_tuple kernel_size, utils::my_tuple stride, utils::my_tuple padding, utils::my_tuple dilation, bool ceil_mode, bool return_indices)
        : kernel_size(kernel_size), stride(stride), padding(padding), dilation(dilation),
          ceil_mode(ceil_mode), return_indices(return_indices)

{}

TensorGrad MaxPool2D::forward(TensorGrad x) {
    return functional::max_pool2d(x, this->kernel_size, this->stride, this->padding, this->dilation, this->ceil_mode, this->return_indices);
}

}
}


_NT_REGISTER_LAYER_NAMESPACED_(nt::layers::MaxPool2D, nt__layers__MaxPool2D, kernel_size,
                               stride, padding, dilation, ceil_mode, return_indices)

