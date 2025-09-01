#ifndef NT_FUNCTIONAL_CPU_DROPOUT_H__
#define NT_FUNCTIONAL_CPU_DROPOUT_H__

#include "../../dtype/ArrayVoid.h"
#include "../../dtype/Scalar.h"

namespace nt::functional::cpu{


NEUROTENSOR_API void _dropout2d_(ArrayVoid&, const ArrayVoid&, const int64_t&, const int64_t&);
NEUROTENSOR_API void _dropout3d_(ArrayVoid&, const ArrayVoid&, const int64_t&, const int64_t&, const int64_t&);

}


#endif
