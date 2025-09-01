#ifndef NT_FUNCTIONAL_TENSOR_FILES_RAND_H__
#define NT_FUNCTIONAL_TENSOR_FILES_RAND_H__

#include "../../Tensor.h"
#include "../../dtype/Scalar.h"

namespace nt {
namespace functional {

NEUROTENSOR_API Tensor randint(Scalar, Scalar, SizeRef, DType dt = DType::int32);
NEUROTENSOR_API Tensor rand(Scalar, Scalar, SizeRef, DType dt = DType::Float32);
NEUROTENSOR_API Tensor randn(SizeRef, DType dt = DType::Float);
NEUROTENSOR_API Tensor randbools(SizeRef, double);//the percentage between 0 and 1 that are true

} // namespace functional
} // namespace nt

#endif
