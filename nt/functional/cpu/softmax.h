#ifndef NT_FUNCTIONAL_CPU_SOFTMAX_H__
#define NT_FUNCTIONAL_CPU_SOFTMAX_H__

#include "../../dtype/ArrayVoid.h"
#include "../../dtype/Scalar.h"

namespace nt {
namespace functional {
namespace cpu {

NEUROTENSOR_API void _softmax(ArrayVoid &in, ArrayVoid &out);
NEUROTENSOR_API void _softmax_stable(ArrayVoid &in, ArrayVoid &out, Scalar max);
NEUROTENSOR_API void _dsoftmax(const ArrayVoid &softmax_output, const ArrayVoid &dL_dY,
               ArrayVoid &out);
NEUROTENSOR_API void _gumbel_algorithm_(ArrayVoid& in_o, ArrayVoid& noise, Scalar tau);

} // namespace cpu
} // namespace functional
} // namespace nt

#endif
