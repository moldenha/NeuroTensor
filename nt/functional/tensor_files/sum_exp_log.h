#ifndef NT_FUNCTIONAL_TENSOR_FILES_SUM_EXP_LOG_H__
#define NT_FUNCTIONAL_TENSOR_FILES_SUM_EXP_LOG_H__

#include "../../utils/optional_list.h"
#include "../../Tensor.h"

namespace nt{
namespace functional{

NEUROTENSOR_API Tensor log(const Tensor&);
NEUROTENSOR_API Tensor dlog(const Tensor&);
NEUROTENSOR_API Tensor exp(const Tensor&);
NEUROTENSOR_API Tensor sum(const Tensor&, utils::optional_list list = nullptr, bool keepdim=false);
NEUROTENSOR_API void dsum(const Tensor& grad, Tensor& out, SizeRef summed_shape);
NEUROTENSOR_API Tensor logsumexp(const Tensor&, utils::optional_list list = nullptr, bool keepdim = false);
NEUROTENSOR_API Tensor dlogsumexp(const Tensor& grad, const Tensor& x, utils::optional_list list);

NEUROTENSOR_API Tensor& exp_(Tensor&);
NEUROTENSOR_API Tensor& log_(Tensor&);

}
}

#endif

