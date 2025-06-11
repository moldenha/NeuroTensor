#include "AdaptiveLPPool1D.h"
#include "../functional.h"
#include "../../functional/functional.h"
#include "../../reflection/layer_reflect/layer_registry.hpp"
#include "../../reflection/layer_reflect/reflect_macros.h"
namespace nt{
namespace layers{

AdaptiveLPPool1D::AdaptiveLPPool1D(Scalar power, int64_t output_size)
        : power(power), output_size(output_size)
{}

TensorGrad AdaptiveLPPool1D::forward(TensorGrad x) {
    return functional::adaptive_lp_pool1d(x, this->output_size, this->power);
}

}
}


_NT_REGISTER_LAYER_NAMESPACED_(nt::layers::AdaptiveLPPool1D, nt__layers__AdaptiveLPPool1D, power, output_size)

