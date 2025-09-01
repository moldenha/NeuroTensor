#ifndef NT_FUNCTIONAL_CPU_FUSED_H__
#define NT_FUNCTIONAL_CPU_FUSED_H__

#include "../../dtype/ArrayVoid.h"
#include "../../dtype/Scalar.h"

namespace nt{
namespace functional{
namespace cpu{

//returns c + (a * b);
NEUROTENSOR_API void _fused_multiply_add(ArrayVoid& a, ArrayVoid& b, ArrayVoid& o);
NEUROTENSOR_API void _fused_multiply_add(ArrayVoid& a, Scalar b, ArrayVoid& o);
//returns c += (a * b);
NEUROTENSOR_API void _fused_multiply_add_(ArrayVoid& c, ArrayVoid& a, ArrayVoid& b);
NEUROTENSOR_API void _fused_multiply_add_(ArrayVoid& c, ArrayVoid& a, Scalar b);
//returns c - (a * b);
NEUROTENSOR_API void _fused_multiply_subtract(ArrayVoid& c, ArrayVoid& a, ArrayVoid& b, ArrayVoid& o);
NEUROTENSOR_API void _fused_multiply_subtract(ArrayVoid& c, ArrayVoid& a, Scalar b, ArrayVoid& o);
//returns c -= (a * b);
NEUROTENSOR_API void _fused_multiply_subtract_(ArrayVoid& c, ArrayVoid& a, ArrayVoid& b);
NEUROTENSOR_API void _fused_multiply_subtract_(ArrayVoid& c, ArrayVoid& a, Scalar b);


}
}
}

#endif
