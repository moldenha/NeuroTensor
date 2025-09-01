#ifndef NT_FUNCTIONAL_TENSOR_FILES_FILL_H__
#define NT_FUNCTIONAL_TENSOR_FILES_FILL_H__

#include "../../Tensor.h"
#include "../../dtype/Scalar.h"

namespace nt {
namespace functional {

NEUROTENSOR_API Tensor zeros(SizeRef, DType dt = DType::Float);
NEUROTENSOR_API Tensor zeros_like(const Tensor &);
NEUROTENSOR_API Tensor ones(SizeRef, DType dt = DType::Float);
NEUROTENSOR_API Tensor ones_like(const Tensor &);
NEUROTENSOR_API Tensor nums(SizeRef, Scalar, DType dt = DType::Float);
NEUROTENSOR_API Tensor nums_like(const Tensor& t, Scalar);
NEUROTENSOR_API Tensor arange(typename Tensor::size_value_t total_size, DType dt = DType::Float, Scalar start = 0);
NEUROTENSOR_API Tensor arange(SizeRef, DType dt = DType::Float, Scalar start = 0);
NEUROTENSOR_API Tensor& fill_diagonal_(Tensor&, Scalar);
NEUROTENSOR_API Tensor& fill_(Tensor&, Scalar);
NEUROTENSOR_API Tensor& set_(Tensor&, const Tensor&);

} // namespace functional
} // namespace nt

#endif
