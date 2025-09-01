#ifndef NT_FUNCTIONAL_TENSOR_FILES_RANGES_H__
#define NT_FUNCTIONAL_TENSOR_FILES_RANGES_H__

#include "../../Tensor.h"

namespace nt {
namespace functional {
NEUROTENSOR_API Tensor get_range(const Tensor &t, const range_ &r, size_t idx);
NEUROTENSOR_API Tensor op_range(Tensor t, range_);
NEUROTENSOR_API Tensor op_range(const Tensor &t, std::vector<range_>);
} // namespace functional
} // namespace nt

#endif
