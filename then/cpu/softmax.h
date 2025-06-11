#ifndef __NT_FUNCTIONAL_CPU_SOFTMAX_H__
#define __NT_FUNCTIONAL_CPU_SOFTMAX_H__

#include "../../dtype/ArrayVoid.h"
#include "../../dtype/Scalar.h"

namespace nt {
namespace functional {
namespace cpu {

void _softmax(ArrayVoid &in, ArrayVoid &out);
void _softmax_stable(ArrayVoid &in, ArrayVoid &out, Scalar max);
void _dsoftmax(const ArrayVoid &softmax_output, const ArrayVoid &dL_dY,
               ArrayVoid &out);

} // namespace cpu
} // namespace functional
} // namespace nt

#endif
