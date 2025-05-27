#ifndef __NT_FUNCTIONAL_CPU_MIN_MAX_H__
#define __NT_FUNCTIONAL_CPU_MIN_MAX_H__

#include "../../dtype/ArrayVoid.h"
#include "../../dtype/Scalar.h"

namespace nt{
namespace functional{
namespace cpu{

void _clamp(ArrayVoid& a, Scalar min, Scalar max);
void _min(ArrayVoid& out, std::vector<ArrayVoid>& arrvds);
void _max(ArrayVoid& out, std::vector<ArrayVoid>& arrvds);
Scalar _min_scalar(const ArrayVoid& in, ArrayVoid& indices);
Scalar _max_scalar(const ArrayVoid& in, ArrayVoid& indices);
void _min_strided(const ArrayVoid& in, ArrayVoid& indices, int64_t cols);
void _max_strided(const ArrayVoid& in, ArrayVoid& indices, int64_t cols);

}
}
}

#endif
