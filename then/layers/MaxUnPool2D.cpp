#include "MaxUnPool2D.h"
#include "../functional.h"
#include "../../functional/functional.h"
#include "../../reflection/layer_reflect/layer_registry.hpp"
#include "../../reflection/layer_reflect/reflect_macros.h"
namespace nt{
namespace layers{

MaxUnPool2D::MaxUnPool2D(utils::my_tuple kernel_size, utils::my_tuple stride, utils::my_tuple padding, utils::my_tuple output_size)
        : kernel_size(kernel_size), stride(stride), padding(padding), output_size(output_size)
{}

TensorGrad MaxUnPool2D::forward(TensorGrad x, TensorGrad indices) {
    return functional::max_unpool2d(x, indices, this->kernel_size, this->stride, this->padding, this->output_size);
}

}
}


_NT_REGISTER_LAYER_NAMESPACED_(nt::layers::MaxUnPool2D, nt__layers__MaxUnPool2D, kernel_size,
                               stride, padding, output_size)

