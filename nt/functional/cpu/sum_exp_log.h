#ifndef NT_FUNCTIONAL_CPU_SUM_LOG_EXP_H__
#define NT_FUNCTIONAL_CPU_SUM_LOG_EXP_H__

#include "../../dtype/ArrayVoid.h"
#include "../../dtype/Scalar.h"

namespace nt{
namespace functional{
namespace cpu{


NEUROTENSOR_API void _exp(const ArrayVoid& a, ArrayVoid&);
NEUROTENSOR_API void _log(const ArrayVoid& a, ArrayVoid& out);
NEUROTENSOR_API void _exp_(ArrayVoid& a);
NEUROTENSOR_API void _log_(ArrayVoid& a);
NEUROTENSOR_API Scalar _accumulate(const ArrayVoid& a, Scalar initial);
NEUROTENSOR_API void _sum_every(const ArrayVoid&, ArrayVoid&, int64_t);

}
}
}

#endif
