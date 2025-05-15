#include "AdaptiveMaxPool2D.h"
#include "../functional.h"
#include "../../functional/functional.h"
#include "../../reflection/layer_reflect/layer_registry.hpp"
#include "../../reflection/layer_reflect/reflect_macros.h"
namespace nt{
namespace layers{

AdaptiveMaxPool2D::AdaptiveMaxPool2D(utils::my_tuple output_size, bool return_indices)
        : output_size(output_size), return_indices(return_indices)

{}

TensorGrad AdaptiveMaxPool2D::forward(TensorGrad x) {
    return functional::adaptive_max_pool2d(x, this->output_size, this->return_indices);
}

}
}

_NT_REGISTER_LAYER_NAMESPACED_(nt::layers::AdaptiveMaxPool2D, nt__layers__AdaptiveMaxPool2D, output_size,
                               return_indices)

