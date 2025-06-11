#include "AdaptiveMaxPool1D.h"
#include "../functional.h"
#include "../../functional/functional.h"
#include "../../reflection/layer_reflect/layer_registry.hpp"
#include "../../reflection/layer_reflect/reflect_macros.h"
namespace nt{
namespace layers{

AdaptiveMaxPool1D::AdaptiveMaxPool1D(int64_t output_size, bool return_indices)
        : output_size(output_size), return_indices(return_indices)

{}

TensorGrad AdaptiveMaxPool1D::forward(TensorGrad x) {
    return functional::adaptive_max_pool1d(x, this->output_size, this->return_indices);
}

}
}


_NT_REGISTER_LAYER_NAMESPACED_(nt::layers::AdaptiveMaxPool1D, nt__layers__AdaptiveMaxPool1D, output_size,
                               return_indices)

