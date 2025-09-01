#ifndef NT_FUNCTIONAL_CPU_RAND_H__
#define NT_FUNCTIONAL_CPU_RAND_H__

#include "../../dtype/ArrayVoid.h"
#include "../../dtype/Scalar.h"
#include "../../refs/SizeRef.h"

namespace nt{
namespace functional{
namespace cpu{

NEUROTENSOR_API void rand_(ArrayVoid& output, Scalar upper, Scalar lower);
NEUROTENSOR_API void randint_(ArrayVoid& output, Scalar upper, Scalar lower);

}
}
}

#endif  
