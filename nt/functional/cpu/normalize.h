#ifndef NT_FUNCTIONAL_CPU_NORMALIZE_H__
#define NT_FUNCTIONAL_CPU_NORMALIZE_H__

#include "../../dtype/ArrayVoid.h"
#include "../../dtype/Scalar.h"
#include "../../refs/SizeRef.h"

namespace nt{
namespace functional{
namespace cpu{

NEUROTENSOR_API void xavier_uniform_(ArrayVoid& output, double bound);

}
}
}

#endif  
