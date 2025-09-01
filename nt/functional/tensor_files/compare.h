#ifndef NT_FUNCTIONAL_TENSOR_FILES_COMPARE_H__
#define NT_FUNCTIONAL_TENSOR_FILES_COMPARE_H__

#include "../../Tensor.h"
namespace nt {
namespace functional {

NEUROTENSOR_API Tensor equal(const Tensor &, const Tensor &);
NEUROTENSOR_API Tensor not_equal(const Tensor &, const Tensor &);
NEUROTENSOR_API Tensor less_than(const Tensor &, const Tensor &);
NEUROTENSOR_API Tensor greater_than(const Tensor &, const Tensor &);
NEUROTENSOR_API Tensor less_than_equal(const Tensor &, const Tensor &);
NEUROTENSOR_API Tensor greater_than_equal(const Tensor &, const Tensor &);
NEUROTENSOR_API Tensor and_op(const Tensor &, const Tensor &);
NEUROTENSOR_API Tensor or_op(const Tensor &, const Tensor &);
NEUROTENSOR_API Tensor isnan(const Tensor&);

NEUROTENSOR_API Tensor equal(const Tensor &, Scalar);
NEUROTENSOR_API Tensor not_equal(const Tensor &, Scalar);
NEUROTENSOR_API Tensor less_than(const Tensor &, Scalar);
NEUROTENSOR_API Tensor greater_than(const Tensor &, Scalar);
NEUROTENSOR_API Tensor less_than_equal(const Tensor &, Scalar);
NEUROTENSOR_API Tensor greater_than_equal(const Tensor &, Scalar);

NEUROTENSOR_API bool all(const Tensor &);
NEUROTENSOR_API bool any(const Tensor &);
NEUROTENSOR_API bool none(const Tensor &);
NEUROTENSOR_API Tensor all(const Tensor, int64_t dim);
NEUROTENSOR_API Tensor any(const Tensor, int64_t dim);
NEUROTENSOR_API int64_t amount_of(Tensor, Scalar val = 0);
NEUROTENSOR_API int64_t count(Tensor);

NEUROTENSOR_API Tensor where(Tensor);
NEUROTENSOR_API bool allclose(const Tensor& input, const Tensor& other, Scalar rtol = float(1e-5), Scalar atol = float(1e-8), bool equal_nan = false);
NEUROTENSOR_API Tensor isclose(const Tensor& input, const Tensor& other, Scalar rtol = float(1e-5), Scalar atol = float(1e-8), bool equal_nan = false);

} // namespace functional
} // namespace nt

#endif
