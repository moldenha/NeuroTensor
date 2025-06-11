#include "Linear.h"
#include "../../functional/functional.h"
#include "../functional.h"
#include "../../reflection/layer_reflect/layer_registry.hpp"
#include "../../reflection/layer_reflect/reflect_macros.h"

namespace nt {
namespace layers {

Linear::Linear(int64_t in_channels, int64_t out_channels, bool use_bias)
        :Weight(functional::randn({in_channels, out_channels})), 
        Bias((use_bias ? functional::randn({out_channels}) : Tensor::Null())),
        use_bias(use_bias) 
{
    if(use_bias){
        this->register_parameter("Bias", this->Bias);
        this->Bias.tensor.set_mutability(false);
    }
    this->Weight.tensor.set_mutability(false);
}

TensorGrad Linear::forward(TensorGrad x){
    return functional::linear(x, this->Weight, this->Bias);
}

} // namespace layers
} // namespace nt

_NT_REGISTER_LAYER_NAMESPACED_(nt::layers::Linear, nt__layers__Linear, use_bias,
                               Weight)
