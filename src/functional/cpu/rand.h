#ifndef __NT_FUNCTIONAL_CPU_RAND_H__
#define __NT_FUNCTIONAL_CPU_RAND_H__

#include "../../dtype/ArrayVoid.h"
#include "../../dtype/Scalar.h"
#include "../../refs/SizeRef.h"

namespace nt{
namespace functional{
namespace cpu{

void rand_(ArrayVoid& output, Scalar upper, Scalar lower);
void randint_(ArrayVoid& output, Scalar upper, Scalar lower);

}
}
}

#endif  
