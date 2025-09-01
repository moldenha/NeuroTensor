#ifndef NT_FUNCTIONAL_TENSOR_FILES_REPEAT_H__
#define NT_FUNCTIONAL_TENSOR_FILES_REPEAT_H__

#include "../../Tensor.h"

namespace nt {
namespace functional {

NEUROTENSOR_API Tensor repeat_(const Tensor &, Tensor::size_value_t amt);
NEUROTENSOR_API Tensor repeat_(const Tensor &, Tensor::size_value_t dim,
               Tensor::size_value_t amt);
NEUROTENSOR_API Tensor expand(const Tensor &, SizeRef);
NEUROTENSOR_API Tensor expand_as(const Tensor &, const Tensor &);

} // namespace functional
} // namespace nt

#endif
