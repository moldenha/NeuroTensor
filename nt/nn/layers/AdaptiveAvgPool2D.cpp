#include "AdaptiveAvgPool2D.h"
#include "../functional.h"
#include "../../functional/functional.h"
#include "../../reflection/layer_reflect/layer_registry.hpp"
#include "../../reflection/layer_reflect/reflect_macros.h"
namespace nt{
namespace layers{

AdaptiveAvgPool2D::AdaptiveAvgPool2D(utils::my_tuple output_size)
        : output_size(output_size)
{}

TensorGrad AdaptiveAvgPool2D::forward(TensorGrad x) {
    return functional::adaptive_avg_pool2d(x, this->output_size);
}

}
}


_NT_REGISTER_LAYER_NAMESPACED_(nt::layers::AdaptiveAvgPool2D, nt__layers__AdaptiveAvgPool2D, output_size)

