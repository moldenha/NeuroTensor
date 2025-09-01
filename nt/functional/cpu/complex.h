#ifndef NT_FUNCTIONAL_CPU_COMPLEX_H__
#define NT_FUNCTIONAL_CPU_COMPLEX_H__

#include "../../dtype/ArrayVoid.h"

namespace nt{
namespace functional{
namespace cpu{

NEUROTENSOR_API void _to_complex_from_real(const ArrayVoid&, ArrayVoid&);
NEUROTENSOR_API void _to_complex_from_imag(const ArrayVoid&, ArrayVoid&);

}
}
}

#endif
