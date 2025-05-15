#include "AvgPool3D.h"
#include "../functional.h"
#include "../../functional/functional.h"
#include "../../reflection/layer_reflect/layer_registry.hpp"
#include "../../reflection/layer_reflect/reflect_macros.h"
namespace nt{
namespace layers{

AvgPool3D::AvgPool3D(utils::my_n_tuple<3> kernel_size, utils::my_n_tuple<3> stride, utils::my_n_tuple<3> padding, bool ceil_mode, bool count_include_pad)
        : kernel_size(kernel_size), stride(stride), padding(padding),
          ceil_mode(ceil_mode), count_include_pad(count_include_pad)

{}

TensorGrad AvgPool3D::forward(TensorGrad x) {
    return functional::avg_pool3d(x, this->kernel_size, this->stride, this->padding, this->ceil_mode, this->count_include_pad);
}

}
}


_NT_REGISTER_LAYER_NAMESPACED_(nt::layers::AvgPool3D, nt__layers__AvgPool3D, kernel_size,
                               stride, padding, ceil_mode, count_include_pad)

