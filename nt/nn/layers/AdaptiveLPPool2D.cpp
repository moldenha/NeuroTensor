#include "AdaptiveLPPool2D.h"
#include "../functional.h"
#include "../../functional/functional.h"
#include "../../reflection/layer_reflect/layer_registry.hpp"
#include "../../reflection/layer_reflect/reflect_macros.h"
namespace nt{
namespace layers{

AdaptiveLPPool2D::AdaptiveLPPool2D(Scalar power, utils::my_tuple output_size)
        : power(power), output_size(output_size)
{}

TensorGrad AdaptiveLPPool2D::forward(TensorGrad x) {
    return functional::adaptive_lp_pool2d(x, this->output_size, this->power);
}

}
}


_NT_REGISTER_LAYER_NAMESPACED_(nt::layers::AdaptiveLPPool2D, nt__layers__AdaptiveLPPool2D, power, output_size)

