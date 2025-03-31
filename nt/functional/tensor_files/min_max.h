// these are functions specifically designed for min-max type functions
// this includes clamp, relu, etc
#ifndef __NT_FUNCTIONAL_TENSOR_FILES_MIN_MAX_H__
#define __NT_FUNCTIONAL_TENSOR_FILES_MIN_MAX_H__

#include "../../Tensor.h"
#include "../../dtype/Scalar.h"

#include <optional>
#include <vector>

namespace nt {
namespace functional {

Tensor clamp(const Tensor &x, std::optional<Scalar> min = std::nullopt,
             std::optional<Scalar> max = std::nullopt);
Tensor relu(const Tensor &);
Tensor silu(const Tensor &);
Tensor dsilu(const Tensor &);
Tensor gelu(const Tensor &);
Tensor dgelu(const Tensor &);
Scalar min(std::vector<Scalar>);
Scalar max(std::vector<Scalar>);
Tensor min(std::vector<Tensor>);
Tensor max(std::vector<Tensor>);
Tensor min(std::vector<Tensor>, Scalar);
Tensor max(std::vector<Tensor>, Scalar);

} // namespace functional
} // namespace nt

#include "min_max.hpp"

#endif
