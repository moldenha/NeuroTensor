#ifndef __NT_FUNCTIONAL_CPU_NORMALIZE_H__
#define __NT_FUNCTIONAL_CPU_NORMALIZE_H__

#include "../../dtype/ArrayVoid.h"
#include "../../dtype/Scalar.h"
#include "../../refs/SizeRef.h"

namespace nt{
namespace functional{
namespace cpu{

void xavier_uniform_(ArrayVoid& output, double bound);

}
}
}

#endif  
