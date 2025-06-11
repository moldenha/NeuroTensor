#ifndef __NT_FUNCTIONAL_CPU_UNIQUE_H__
#define __NT_FUNCTIONAL_CPU_UNIQUE_H__

#include "../../Tensor.h"

namespace nt{
namespace functional{
namespace cpu{

Tensor _unique(Tensor input, int64_t dim, bool return_sorted, bool return_indices);

}
}
}

#endif
