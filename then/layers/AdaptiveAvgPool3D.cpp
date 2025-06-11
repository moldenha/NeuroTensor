#include "AdaptiveAvgPool3D.h"
#include "../functional.h"
#include "../../functional/functional.h"
#include "../../reflection/layer_reflect/layer_registry.hpp"
#include "../../reflection/layer_reflect/reflect_macros.h"
namespace nt{
namespace layers{

AdaptiveAvgPool3D::AdaptiveAvgPool3D(utils::my_n_tuple<3> output_size)
        : output_size(output_size)
{}

TensorGrad AdaptiveAvgPool3D::forward(TensorGrad x) {
    return functional::adaptive_avg_pool3d(x, this->output_size);
}

}
}


_NT_REGISTER_LAYER_NAMESPACED_(nt::layers::AdaptiveAvgPool3D, nt__layers__AdaptiveAvgPool3D, output_size)

