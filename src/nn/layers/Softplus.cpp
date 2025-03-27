#include "Softplus.h"
#include "../../functional/functional.h"
#include "../functional.h"
#include "../layer_reflect/layer_registry.hpp"
#include "../layer_reflect/reflect_macros.h"

namespace nt {
namespace layers {

Softplus::Softplus(Scalar beta, Scalar threshold)
    : beta(beta), threshold(threshold) {}

TensorGrad Softplus::forward(TensorGrad x) {
    return functional::softplus(x, beta, threshold);
}

} // namespace layers
} // namespace nt

_NT_REGISTER_LAYER_NAMESPACED_(nt::layers::Softplus, nt__layers__Softplus, beta,
                               threshold)
