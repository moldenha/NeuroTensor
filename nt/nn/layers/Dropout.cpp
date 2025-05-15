#include "Dropout.h"
#include "../../functional/functional.h"
#include "../functional.h"
#include "../../reflection/layer_reflect/layer_registry.hpp"
#include "../../reflection/layer_reflect/reflect_macros.h"

namespace nt {
namespace layers {

Dropout::Dropout(Scalar s) : p(s.to<double>()) {
    utils::throw_exception(p >= 0 && p <= 1,
                           "Expected p for dropout to be in [0, 1] but got $",
                           p);
}

Dropout::Dropout(double s) : p(s) {
    utils::throw_exception(p >= 0 && p <= 1,
                           "Expected p for dropout to be in [0, 1] but got $",
                           p);
}

TensorGrad Dropout::forward(TensorGrad x) { return functional::dropout(x, p); }

} // namespace layers
} // namespace nt

_NT_REGISTER_LAYER_NAMESPACED_(nt::layers::Dropout, nt__layers__Dropout, p)
