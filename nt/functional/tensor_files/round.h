#ifndef NT_FUNCTIONAL_TENSOR_FILES_ROUND_H__
#define NT_FUNCTIONAL_TENSOR_FILES_ROUND_H__

#include "../../Tensor.h"
#include "../../dtype/Scalar.h"

namespace nt {
namespace functional {

NEUROTENSOR_API Tensor round(const Tensor&, int64_t decimals = 0);
NEUROTENSOR_API Tensor floor(const Tensor&);
NEUROTENSOR_API Tensor ceil(const Tensor&);
NEUROTENSOR_API Tensor trunc(const Tensor&);

} // namespace functional
} // namespace nt

#endif
