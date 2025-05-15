#ifndef __NT_FUNCTIONAL_TENSOR_FILES_SOFTMAX_H__
#define __NT_FUNCTIONAL_TENSOR_FILES_SOFTMAX_H__

#include "../../Tensor.h"
namespace nt {
namespace functional {

void softmax_(Tensor &);
void softmax_(Tensor &, typename SizeRef::value_type);
void softmax_stable_(Tensor &);
void softmax_stable_(Tensor &, typename SizeRef::value_type);
Tensor softmax(Tensor);
Tensor softmax(Tensor, typename SizeRef::value_type);
Tensor softmax_stable(Tensor);
Tensor softmax_stable(Tensor, typename SizeRef::value_type);

Tensor dsoftmax(const Tensor& dy, const Tensor& last_softmax);
Tensor dsoftmax(const Tensor& dy, const Tensor& last_softmax, typename SizeRef::value_type dim);
} // namespace functional
} // namespace nt

#endif
