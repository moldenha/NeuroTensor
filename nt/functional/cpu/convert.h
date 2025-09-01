#ifndef NT_FUNCTIONAL_CPU_CONVERT_H__
#define NT_FUNCTIONAL_CPU_CONVERT_H__

#include "../../dtype/ArrayVoid.h"

namespace nt{
namespace functional{
namespace cpu{

NEUROTENSOR_API void _convert(const ArrayVoid&, ArrayVoid&);
NEUROTENSOR_API void _floating_(ArrayVoid&);
NEUROTENSOR_API void _complex_(ArrayVoid&);
NEUROTENSOR_API void _integer_(ArrayVoid&);
NEUROTENSOR_API void _unsigned_(ArrayVoid&);

}
}
}

#endif
