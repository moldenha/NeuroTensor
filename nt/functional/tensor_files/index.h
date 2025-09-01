#ifndef NT_FUNCTIONAL_TENSOR_FILES_INDEX_H__
#define NT_FUNCTIONAL_TENSOR_FILES_INDEX_H__

#include "../../Tensor.h"
#include "../../dtype/Scalar.h"

namespace nt {
namespace functional {
// this is a functional of the tensor [] operator

NEUROTENSOR_API Tensor at(const Tensor &, Tensor::size_value_t);
NEUROTENSOR_API Tensor at(const Tensor &, const Tensor &);
NEUROTENSOR_API Tensor at(const Tensor &, std::vector<Tensor::size_value_t>);
NEUROTENSOR_API Tensor at_tensor_split(const Tensor &, const Tensor &, Tensor::size_value_t);
NEUROTENSOR_API Tensor &at_tensor_split(const Tensor &, const Tensor &, Tensor::size_value_t,
                        Tensor &);
NEUROTENSOR_API Tensor index_except(const Tensor &, int64_t, Tensor::size_value_t);
NEUROTENSOR_API Tensor index_select(Tensor input, int64_t dim, Tensor index);
NEUROTENSOR_API Tensor select(Tensor input, Tensor::size_value_t dim, Tensor::size_value_t index);


} // namespace functional
} // namespace nt

#endif
