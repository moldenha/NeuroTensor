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

}
}
}

#endif
