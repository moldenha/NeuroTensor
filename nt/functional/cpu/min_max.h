#ifndef NT_FUNCTIONAL_CPU_MIN_MAX_H__
#define NT_FUNCTIONAL_CPU_MIN_MAX_H__

#include "../../dtype/ArrayVoid.h"
#include "../../dtype/Scalar.h"

namespace nt{
namespace functional{
namespace cpu{

NEUROTENSOR_API void _clamp(ArrayVoid& a, Scalar min, Scalar max);
NEUROTENSOR_API void _min(ArrayVoid& out, std::vector<ArrayVoid>& arrvds);
NEUROTENSOR_API void _max(ArrayVoid& out, std::vector<ArrayVoid>& arrvds);
NEUROTENSOR_API Scalar _min_scalar(const ArrayVoid& in, ArrayVoid& indices);
NEUROTENSOR_API Scalar _max_scalar(const ArrayVoid& in, ArrayVoid& indices);
NEUROTENSOR_API void _min_strided(const ArrayVoid& in, ArrayVoid& indices, int64_t cols);
NEUROTENSOR_API void _max_strided(const ArrayVoid& in, ArrayVoid& indices, int64_t cols);
// x < max
NEUROTENSOR_API void _clamp_above(ArrayVoid& a, Scalar max);
// min < x
NEUROTENSOR_API void _clamp_below(ArrayVoid& a, Scalar min);
}
}
}

#endif
