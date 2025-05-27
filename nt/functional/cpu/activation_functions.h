#ifndef __NT_FUNCTIONAL_CPU_ACTIVATION_FUNCTIONS_H__
#define __NT_FUNCTIONAL_CPU_ACTIVATION_FUNCTIONS_H__

#include "../../dtype/ArrayVoid.h"
#include "../../dtype/Scalar.h"

namespace nt{
namespace functional{
namespace cpu{

void _sigmoid(const ArrayVoid&, ArrayVoid&);
void _dsigmoid(const ArrayVoid&, ArrayVoid&, const bool&);
void _sqrt(const ArrayVoid&, ArrayVoid&);
void _invsqrt(const ArrayVoid&, ArrayVoid&);
void _dsqrt(const ArrayVoid&, ArrayVoid&);
void _dinvsqrt(const ArrayVoid&, ArrayVoid&);
void _pow(const ArrayVoid&, ArrayVoid&, Scalar);
void _abs(const ArrayVoid&, ArrayVoid&);
void _silu(const ArrayVoid&, ArrayVoid&);
void _dsilu(const ArrayVoid&, ArrayVoid&);
void _gelu(const ArrayVoid&, ArrayVoid&);
void _dgelu(const ArrayVoid&, ArrayVoid&);

void _sigmoid_(ArrayVoid&);
void _dsigmoid_(ArrayVoid&, const bool&);
void _sqrt_(ArrayVoid&);
void _invsqrt_(ArrayVoid&);
void _dsqrt_(ArrayVoid&);
void _dinvsqrt_(ArrayVoid&);
void _pow_(ArrayVoid&, Scalar);
void _abs_(ArrayVoid&);
void _silu_(ArrayVoid&);
void _dsilu_(ArrayVoid&);
void _gelu_(ArrayVoid&);
void _dgelu_(ArrayVoid&);


}
}
}

#endif
