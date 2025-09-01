#ifndef NT_FUNCTIONAL_TENSOR_FILES_SPLIT_H__
#define NT_FUNCTIONAL_TENSOR_FILES_SPLIT_H__

#include "../../Tensor.h"
#include "../../dtype/Scalar.h"

namespace nt {
namespace functional {

NEUROTENSOR_API Tensor split(const Tensor& input, int64_t dim, utils::optional_list splits = nullptr);
NEUROTENSOR_API Tensor chunk(const Tensor& input, typename Tensor::size_value_t chunks,
             int64_t dim = 0); // splits into that many chunks
} // namespace functional
} // namespace nt

#endif
