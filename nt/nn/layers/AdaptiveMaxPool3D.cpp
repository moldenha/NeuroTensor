#include "AdaptiveMaxPool3D.h"
#include "../functional.h"
#include "../../functional/functional.h"
#include "../../reflection/layer_reflect/layer_registry.hpp"
#include "../../reflection/layer_reflect/reflect_macros.h"
namespace nt{
namespace layers{

AdaptiveMaxPool3D::AdaptiveMaxPool3D(utils::my_n_tuple<3> output_size, bool return_indices)
        : output_size(output_size), return_indices(return_indices)

{}

TensorGrad AdaptiveMaxPool3D::forward(TensorGrad x) {
    return functional::adaptive_max_pool3d(x, this->output_size, this->return_indices);
}

}
}

_NT_REGISTER_LAYER_NAMESPACED_(nt::layers::AdaptiveMaxPool3D, nt__layers__AdaptiveMaxPool3D, output_size,
                               return_indices)

