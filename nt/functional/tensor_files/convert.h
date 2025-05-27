#ifndef __NT_FUNCTIONAL_TENSOR_FILES_CONVERT_H__
#define __NT_FUNCTIONAL_TENSOR_FILES_CONVERT_H__

#include "../../Tensor.h"
namespace nt {
namespace functional {

Tensor to(const Tensor&, DType);
Tensor& floating_(Tensor&);
Tensor& complex_(Tensor&);
Tensor& integer_(Tensor&);
Tensor& unsigned_(Tensor&);



} // namespace functional
} // namespace nt

#endif
