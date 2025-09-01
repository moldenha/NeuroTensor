#ifndef NT_FUNCTIONAL_TENSOR_FILES_CONVERT_H__
#define NT_FUNCTIONAL_TENSOR_FILES_CONVERT_H__

#include "../../Tensor.h"
namespace nt {
namespace functional {

NEUROTENSOR_API Tensor to(const Tensor&, DType);
NEUROTENSOR_API Tensor& floating_(Tensor&);
NEUROTENSOR_API Tensor& complex_(Tensor&);
NEUROTENSOR_API Tensor& integer_(Tensor&);
NEUROTENSOR_API Tensor& unsigned_(Tensor&);



} // namespace functional
} // namespace nt

#endif
