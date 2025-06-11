#include "LPPool2D.h"
#include "../functional.h"
#include "../../functional/functional.h"
#include "../../reflection/layer_reflect/layer_registry.hpp"
#include "../../reflection/layer_reflect/reflect_macros.h"
namespace nt{
namespace layers{

LPPool2D::LPPool2D(Scalar power, utils::my_tuple kernel_size, utils::my_tuple stride, bool ceil_mode)
        : power(power), kernel_size(kernel_size), stride(stride),
        ceil_mode(ceil_mode)

{}

TensorGrad LPPool2D::forward(TensorGrad x) {
    return functional::lp_pool2d(x, this->power, this->kernel_size, this->stride, this->ceil_mode);
}

}
}


_NT_REGISTER_LAYER_NAMESPACED_(nt::layers::LPPool2D, nt__layers__LPPool2D, power, kernel_size,
                               stride, ceil_mode)

