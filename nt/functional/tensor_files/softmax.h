#ifndef NT_FUNCTIONAL_TENSOR_FILES_SOFTMAX_H__
#define NT_FUNCTIONAL_TENSOR_FILES_SOFTMAX_H__

#include "../../Tensor.h"
namespace nt {
namespace functional {

NEUROTENSOR_API void softmax_(Tensor &);
NEUROTENSOR_API void softmax_(Tensor &, typename SizeRef::value_type);
NEUROTENSOR_API void softmax_stable_(Tensor &);
NEUROTENSOR_API void softmax_stable_(Tensor &, typename SizeRef::value_type);
NEUROTENSOR_API Tensor softmax(Tensor);
NEUROTENSOR_API Tensor softmax(Tensor, typename SizeRef::value_type);
NEUROTENSOR_API Tensor softmax_stable(Tensor);
NEUROTENSOR_API Tensor softmax_stable(Tensor, typename SizeRef::value_type);

NEUROTENSOR_API Tensor gumbel_softmax(const Tensor&, Scalar tau, bool hard, int64_t dim = -1, bool stable = true);

NEUROTENSOR_API Tensor dsoftmax(const Tensor &dy, const Tensor &last_softmax);
NEUROTENSOR_API Tensor dsoftmax(const Tensor &dy, const Tensor &last_softmax,
                typename SizeRef::value_type dim);
} // namespace functional
} // namespace nt

#endif
