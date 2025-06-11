#ifndef __NT_FUNCTIONAL_TENSOR_FILES_FILL_H__
#define __NT_FUNCTIONAL_TENSOR_FILES_FILL_H__

#include "../../Tensor.h"
#include "../../dtype/Scalar.h"

namespace nt {
namespace functional {
Tensor zeros(SizeRef, DType dt = DType::Float);
Tensor zeros_like(const Tensor &);
Tensor ones(SizeRef, DType dt = DType::Float);
Tensor ones_like(const Tensor &);
Tensor nums(SizeRef, const Scalar, DType dt = DType::Float);
Tensor nums_like(const Tensor& t, const Scalar);
Tensor arange(typename Tensor::size_value_t total_size, DType dt = DType::Float, Scalar start = 0);
Tensor arange(SizeRef, DType dt = DType::Float, Scalar start = 0);
Tensor& fill_diagonal_(Tensor&, Scalar);
Tensor& fill_(Tensor&, Scalar);
Tensor& set_(Tensor&, const Tensor&);

} // namespace functional
} // namespace nt

#endif
