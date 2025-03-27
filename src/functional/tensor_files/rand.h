#ifndef __NT_FUNCTIONAL_TENSOR_FILES_RAND_H__
#define __NT_FUNCTIONAL_TENSOR_FILES_RAND_H__

#include "../../Tensor.h"
#include "../../dtype/Scalar.h"

namespace nt {
namespace functional {

Tensor randint(Scalar, Scalar, SizeRef, DType dt = DType::int32);
Tensor rand(Scalar, Scalar, SizeRef, DType dt = DType::Float32);
Tensor randn(SizeRef, DType dt = DType::Float);
Tensor randbools(SizeRef, double);//the percentage between 0 and 1 that are true

} // namespace functional
} // namespace nt

#endif
