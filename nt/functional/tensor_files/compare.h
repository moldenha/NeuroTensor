#ifndef __NT_FUNCTIONAL_TENSOR_FILES_COMPARE_H__
#define __NT_FUNCTIONAL_TENSOR_FILES_COMPARE_H__

#include "../../Tensor.h"
namespace nt {
namespace functional {

Tensor equal(const Tensor &, const Tensor &);
Tensor not_equal(const Tensor &, const Tensor &);
Tensor less_than(const Tensor &, const Tensor &);
Tensor greater_than(const Tensor &, const Tensor &);
Tensor less_than_equal(const Tensor &, const Tensor &);
Tensor greater_than_equal(const Tensor &, const Tensor &);
Tensor and_op(const Tensor &, const Tensor &);
Tensor or_op(const Tensor &, const Tensor &);

Tensor equal(const Tensor &, Scalar);
Tensor not_equal(const Tensor &, Scalar);
Tensor less_than(const Tensor &, Scalar);
Tensor greater_than(const Tensor &, Scalar);
Tensor less_than_equal(const Tensor &, Scalar);
Tensor greater_than_equal(const Tensor &, Scalar);

bool all(const Tensor &);
bool any(const Tensor &);
bool none(const Tensor &);
int64_t amount_of(Tensor, Scalar val = 0);
int64_t count(Tensor);

Tensor where(Tensor);

} // namespace functional
} // namespace nt

#endif
