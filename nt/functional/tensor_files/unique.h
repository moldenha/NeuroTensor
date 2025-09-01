#ifndef NT_FUNCTIONAL_TENSOR_FILES_UNIQUE_H__
#define NT_FUNCTIONAL_TENSOR_FILES_UNIQUE_H__

#include "../../Tensor.h"
#include "../../dtype/Scalar.h"
#include <optional>

namespace nt {
namespace functional {

NEUROTENSOR_API Tensor unique(Tensor, std::optional<int64_t> dim = std::nullopt, bool return_unique = true,
              bool return_indices = true);
// returns Tensor(unique, indices) (depending on return_unique and
// return_indices)

} // namespace functional
} // namespace nt

#endif
