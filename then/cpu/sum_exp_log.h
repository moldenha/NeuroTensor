#ifndef __NT_FUNCTIONAL_CPU_SUM_LOG_EXP_H__
#define __NT_FUNCTIONAL_CPU_SUM_LOG_EXP_H__

#include "../../dtype/ArrayVoid.h"
#include "../../dtype/Scalar.h"

namespace nt{
namespace functional{
namespace cpu{


void _exp(const ArrayVoid& a, ArrayVoid&);
void _log(const ArrayVoid& a, ArrayVoid& out);
void _exp_(ArrayVoid& a);
void _log_(ArrayVoid& a);
Scalar _accumulate(const ArrayVoid& a, Scalar initial);
void _sum_every(const ArrayVoid&, ArrayVoid&, int64_t);

}
}
}

#endif
