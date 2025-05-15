#ifndef __NT_FUNCTIONAL_TENSOR_FILES_SUM_EXP_LOG_H__
#define __NT_FUNCTIONAL_TENSOR_FILES_SUM_EXP_LOG_H__

#include "../../utils/optional_list.h"
#include "../../Tensor.h"

namespace nt{
namespace functional{

Tensor log(Tensor);
Tensor dlog(Tensor);
Tensor exp(Tensor);
Tensor sum(Tensor, utils::optional_list list = nullptr, bool keepdim=false);
void dsum(Tensor grad, Tensor& out, SizeRef summed_shape);
Tensor logsumexp(Tensor, utils::optional_list list = nullptr, bool keepdim = false);
Tensor dlogsumexp(Tensor grad, Tensor x, utils::optional_list list);

}
}

#endif

