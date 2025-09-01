#ifndef NT_FUNCTIONAL_TENSOR_FILES_COMPLEX_H__
#define NT_FUNCTIONAL_TENSOR_FILES_COMPLEX_H__

#include "../../Tensor.h"
#include "../../dtype/Scalar.h"

namespace nt {
namespace functional {

NEUROTENSOR_API Tensor real(const Tensor&);
NEUROTENSOR_API Tensor imag(const Tensor&);
NEUROTENSOR_API Tensor to_complex_from_real(const Tensor&);
NEUROTENSOR_API Tensor to_complex_from_imag(const Tensor&);

} // namespace functional
} // namespace nt

#endif
