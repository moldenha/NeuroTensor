#include "SymmetricBilinear.h"
#include "../../functional/functional.h"
#include "../functional.h"
#include "../../reflection/layer_reflect/layer_registry.hpp"
#include "../../reflection/layer_reflect/reflect_macros.h"
namespace nt {
namespace layers {

SymmetricBilinear::SymmetricBilinear(int64_t input_size, int64_t hidden_size,
                                     bool use_bias)
    : W1(functional::randn({input_size, input_size})),
      W2(functional::randn({hidden_size, hidden_size})),
      Bias(use_bias ? functional::randn({input_size}) : Tensor::Null()),
      use_bias(use_bias) 
{
    if(use_bias){
        this->register_parameter("Bias", Bias);
        Bias.detach().set_mutability(false);
    }
    W1.detach().set_mutability(false);
    W2.detach().set_mutability(false);
}

TensorGrad SymmetricBilinear::forward(TensorGrad x) {
    utils::throw_exception(x.dims() >= 2,
                            "Expected input to symmetric bilinear layer ($) dims to be greater than or equal to 2", x.shape());
    utils::throw_exception(x.shape()[-1] == x.shape()[-2],
                            "Expected input to symmetric bilinear layer ($) to be a square matrix (can be batched)", x.shape());
    const int64_t& input_size = this->W1.shape()[0];
    utils::throw_exception(x.shape()[-1] == input_size,
                            "Expected input to symmetric bilinear layer ($) matrix size to be specified input size ($)", 
                            x.shape(), input_size);
    TensorGrad out = functional::symmetric_bilinear(x, this->W1, this->W2);
    if(!this->use_bias) return out;
    TensorGrad B = this->Bias.view(-1, 1) * this->Bias.view(1, -1);
    B.detach().set_mutability(false);
    return out + B;

    

}

} // namespace layers
} // namespace nt

_NT_REGISTER_LAYER_NAMESPACED_(nt::layers::SymmetricBilinear,
                               nt__layers__SymmetricBilinear, use_bias, W1, W2)
