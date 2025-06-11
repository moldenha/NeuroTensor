#include "MaxUnPool1D.h"
#include "../functional.h"
#include "../../functional/functional.h"
#include "../../reflection/layer_reflect/layer_registry.hpp"
#include "../../reflection/layer_reflect/reflect_macros.h"
namespace nt{
namespace layers{

MaxUnPool1D::MaxUnPool1D(int64_t kernel_size, int64_t stride, int64_t padding, int64_t output_size)
        : kernel_size(kernel_size), stride(stride), padding(padding), output_size(output_size)
{}

TensorGrad MaxUnPool1D::forward(TensorGrad x, TensorGrad indices) {
    return functional::max_unpool1d(x, indices, this->kernel_size, this->stride, this->padding, this->output_size);
}

}
}


_NT_REGISTER_LAYER_NAMESPACED_(nt::layers::MaxUnPool1D, nt__layers__MaxUnPool1D, kernel_size,
                               stride, padding, output_size)

