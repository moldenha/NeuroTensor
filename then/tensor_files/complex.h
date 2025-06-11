#ifndef __NT_FUNCTIONAL_TENSOR_FILES_COMPLEX_H__
#define __NT_FUNCTIONAL_TENSOR_FILES_COMPLEX_H__

#include "../../Tensor.h"
#include "../../dtype/Scalar.h"

namespace nt {
namespace functional {
Tensor real(const Tensor&);
Tensor imag(const Tensor&);
Tensor to_complex_from_real(const Tensor&);
Tensor to_complex_from_imag(const Tensor&);

} // namespace functional
} // namespace nt

#endif
