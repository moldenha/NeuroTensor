#ifndef __NT_FUNCTIONAL_CPU_SUM_LOG_EXP_H__
#define __NT_FUNCTIONAL_CPU_SUM_LOG_EXP_H__

#include "../../dtype/ArrayVoid.h"
#include "../../dtype/Scalar.h"

namespace nt{
namespace functional{
namespace cpu{


void _exp(ArrayVoid& a, ArrayVoid&);
void _log(const ArrayVoid& a, ArrayVoid& out);
Scalar _accumulate(const ArrayVoid& a, Scalar initial);
void _sum_every(const ArrayVoid&, ArrayVoid&, int64_t);

}
}
}

#endif
