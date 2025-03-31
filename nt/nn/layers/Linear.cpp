#include "Linear.h"
#include "../../functional/functional.h"
#include "../functional.h"
#include "../layer_reflect/layer_registry.hpp"
#include "../layer_reflect/reflect_macros.h"

namespace nt {
namespace layers {

Linear::Linear(int64_t in_dims, int64_t out_dims, bool use_bias)
        :Weight(functional::randn({in_dims, out_dims})), 
        Bias((use_bias ? functional::randn({out_dims}) : Tensor::Null())),
        use_bias(use_bias) 
{}

TensorGrad Linear::forward(TensorGrad x){
    TensorGrad out = functional::matmult(x, Weight);
    if (use_bias) {
        out += Bias;
    }
    return std::move(out);
}

} // namespace layers
} // namespace nt

_NT_REGISTER_LAYER_NAMESPACED_(nt::layers::Linear, nt__layers__Linear, use_bias,
                               Weight, Bias)
