#ifndef __NT_FUNCTIONAL_CPU_FUSED_H__
#define __NT_FUNCTIONAL_CPU_FUSED_H__

#include "../../dtype/ArrayVoid.h"
#include "../../dtype/Scalar.h"

namespace nt{
namespace functional{
namespace cpu{

//returns c + (a * b);
void _fused_multiply_add(ArrayVoid& a, ArrayVoid& b, ArrayVoid& o);
void _fused_multiply_add(ArrayVoid& a, Scalar b, ArrayVoid& o);
//returns c += (a * b);
void _fused_multiply_add_(ArrayVoid& c, ArrayVoid& a, ArrayVoid& b);
void _fused_multiply_add_(ArrayVoid& c, ArrayVoid& a, Scalar b);
//returns c - (a * b);
void _fused_multiply_subtract(ArrayVoid& c, ArrayVoid& a, ArrayVoid& b, ArrayVoid& o);
void _fused_multiply_subtract(ArrayVoid& c, ArrayVoid& a, Scalar b, ArrayVoid& o);
//returns c -= (a * b);
void _fused_multiply_subtract_(ArrayVoid& c, ArrayVoid& a, ArrayVoid& b);
void _fused_multiply_subtract_(ArrayVoid& c, ArrayVoid& a, Scalar b);


}
}
}

#endif
