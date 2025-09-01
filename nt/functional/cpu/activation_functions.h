#ifndef NT_FUNCTIONAL_CPU_ACTIVATION_FUNCTIONS_H__
#define NT_FUNCTIONAL_CPU_ACTIVATION_FUNCTIONS_H__

#include "../../dtype/ArrayVoid.h"
#include "../../dtype/Scalar.h"

namespace nt{
namespace functional{
namespace cpu{

NEUROTENSOR_API void _sigmoid(const ArrayVoid&, ArrayVoid&);
NEUROTENSOR_API void _dsigmoid(const ArrayVoid&, ArrayVoid&, const bool&);
NEUROTENSOR_API void _sqrt(const ArrayVoid&, ArrayVoid&);
NEUROTENSOR_API void _invsqrt(const ArrayVoid&, ArrayVoid&);
NEUROTENSOR_API void _dsqrt(const ArrayVoid&, ArrayVoid&);
NEUROTENSOR_API void _dinvsqrt(const ArrayVoid&, ArrayVoid&);
NEUROTENSOR_API void _pow(const ArrayVoid&, ArrayVoid&, Scalar);
NEUROTENSOR_API void _abs(const ArrayVoid&, ArrayVoid&);
NEUROTENSOR_API void _silu(const ArrayVoid&, ArrayVoid&);
NEUROTENSOR_API void _dsilu(const ArrayVoid&, ArrayVoid&);
NEUROTENSOR_API void _gelu(const ArrayVoid&, ArrayVoid&);
NEUROTENSOR_API void _dgelu(const ArrayVoid&, ArrayVoid&);

NEUROTENSOR_API void _sigmoid_(ArrayVoid&);
NEUROTENSOR_API void _dsigmoid_(ArrayVoid&, const bool&);
NEUROTENSOR_API void _sqrt_(ArrayVoid&);
NEUROTENSOR_API void _invsqrt_(ArrayVoid&);
NEUROTENSOR_API void _dsqrt_(ArrayVoid&);
NEUROTENSOR_API void _dinvsqrt_(ArrayVoid&);
NEUROTENSOR_API void _pow_(ArrayVoid&, Scalar);
NEUROTENSOR_API void _abs_(ArrayVoid&);
NEUROTENSOR_API void _silu_(ArrayVoid&);
NEUROTENSOR_API void _dsilu_(ArrayVoid&);
NEUROTENSOR_API void _gelu_(ArrayVoid&);
NEUROTENSOR_API void _dgelu_(ArrayVoid&);


}
}
}

#endif
