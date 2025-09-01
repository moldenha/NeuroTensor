#ifndef NT_FUNCTIONAL_CPU_ROUND_H__
#define NT_FUNCTIONAL_CPU_ROUND_H__

#include "../../dtype/ArrayVoid.h"

namespace nt{
namespace functional{
namespace cpu{

NEUROTENSOR_API void _round(const ArrayVoid&, ArrayVoid&);
NEUROTENSOR_API void _round_decimal(const ArrayVoid&, ArrayVoid&, int64_t);
NEUROTENSOR_API void _trunc(const ArrayVoid&, ArrayVoid&);
NEUROTENSOR_API void _floor(const ArrayVoid&, ArrayVoid&);
NEUROTENSOR_API void _ceil(const ArrayVoid&, ArrayVoid&);

}
}
}

#endif
