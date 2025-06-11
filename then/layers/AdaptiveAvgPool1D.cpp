#include "AdaptiveAvgPool1D.h"
#include "../functional.h"
#include "../../functional/functional.h"
#include "../../reflection/layer_reflect/layer_registry.hpp"
#include "../../reflection/layer_reflect/reflect_macros.h"
namespace nt{
namespace layers{

AdaptiveAvgPool1D::AdaptiveAvgPool1D(int64_t output_size)
        : output_size(output_size)
{}

TensorGrad AdaptiveAvgPool1D::forward(TensorGrad x) {
    return functional::adaptive_avg_pool1d(x, this->output_size);
}

}
}


_NT_REGISTER_LAYER_NAMESPACED_(nt::layers::AdaptiveAvgPool1D, nt__layers__AdaptiveAvgPool1D, output_size)

