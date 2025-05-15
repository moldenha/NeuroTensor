#include "MaxUnPool3D.h"
#include "../functional.h"
#include "../../functional/functional.h"
#include "../../reflection/layer_reflect/layer_registry.hpp"
#include "../../reflection/layer_reflect/reflect_macros.h"
namespace nt{
namespace layers{

MaxUnPool3D::MaxUnPool3D(utils::my_n_tuple<3> kernel_size, utils::my_n_tuple<3> stride, utils::my_n_tuple<3> padding, utils::my_n_tuple<3> output_size)
        : kernel_size(kernel_size), stride(stride), padding(padding), output_size(output_size)
{}

TensorGrad MaxUnPool3D::forward(TensorGrad x, TensorGrad indices) {
    return functional::max_unpool3d(x, indices, this->kernel_size, this->stride, this->padding, this->output_size);
}

}
}


_NT_REGISTER_LAYER_NAMESPACED_(nt::layers::MaxUnPool3D, nt__layers__MaxUnPool3D, kernel_size,
                               stride, padding, output_size)

