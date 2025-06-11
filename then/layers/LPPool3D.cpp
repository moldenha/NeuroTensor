#include "LPPool3D.h"
#include "../functional.h"
#include "../../functional/functional.h"
#include "../../reflection/layer_reflect/layer_registry.hpp"
#include "../../reflection/layer_reflect/reflect_macros.h"
namespace nt{
namespace layers{

LPPool3D::LPPool3D(Scalar power, utils::my_n_tuple<3> kernel_size, utils::my_n_tuple<3> stride, bool ceil_mode)
        : power(power), kernel_size(kernel_size), stride(stride),
        ceil_mode(ceil_mode)

{}

TensorGrad LPPool3D::forward(TensorGrad x) {
    return functional::lp_pool3d(x, this->power, this->kernel_size, this->stride, this->ceil_mode);
}

}
}


_NT_REGISTER_LAYER_NAMESPACED_(nt::layers::LPPool3D, nt__layers__LPPool3D, power, kernel_size,
                               stride, ceil_mode)

