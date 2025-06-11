#include "MaxPool3D.h"
#include "../functional.h"
#include "../../functional/functional.h"
#include "../../reflection/layer_reflect/layer_registry.hpp"
#include "../../reflection/layer_reflect/reflect_macros.h"

namespace nt{
namespace layers{

MaxPool3D::MaxPool3D(utils::my_n_tuple<3> kernel_size, utils::my_n_tuple<3> stride, utils::my_n_tuple<3> padding, utils::my_n_tuple<3> dilation, bool ceil_mode, bool return_indices)
        : kernel_size(kernel_size), stride(stride), padding(padding), dilation(dilation),
          ceil_mode(ceil_mode), return_indices(return_indices)

{}

TensorGrad MaxPool3D::forward(TensorGrad x) {
    return functional::max_pool3d(x, this->kernel_size, this->stride, this->padding, this->dilation, this->ceil_mode, this->return_indices);
}

}
}


_NT_REGISTER_LAYER_NAMESPACED_(nt::layers::MaxPool3D, nt__layers__MaxPool3D, kernel_size,
                               stride, padding, dilation, ceil_mode, return_indices)

