#ifndef NT_FUNCTIONAL_CPU_FILL_H__
#define NT_FUNCTIONAL_CPU_FILL_H__

#include "../../dtype/ArrayVoid.h"
#include "../../dtype/Scalar.h"

namespace nt{
namespace functional{
namespace cpu{

NEUROTENSOR_API void _fill_diagonal_(ArrayVoid& arr, Scalar s, const int64_t& batches, const int64_t& rows, const int64_t& cols);
NEUROTENSOR_API void _fill_scalar_(ArrayVoid& arr, Scalar s);
NEUROTENSOR_API void _set_(ArrayVoid&, const ArrayVoid&);
NEUROTENSOR_API void _iota_(ArrayVoid& arr, Scalar start); //arange

}
}
}


#endif
