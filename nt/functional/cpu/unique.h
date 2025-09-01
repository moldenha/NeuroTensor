#ifndef NT_FUNCTIONAL_CPU_UNIQUE_H__
#define NT_FUNCTIONAL_CPU_UNIQUE_H__

#include "../../Tensor.h"

namespace nt{
namespace functional{
namespace cpu{

NEUROTENSOR_API Tensor _unique_vals_only(const Tensor& input, bool return_sorted, bool return_indices);
NEUROTENSOR_API Tensor _unique(const Tensor& _input, int64_t dim, bool return_sorted, bool return_indices);

}
}
}

#endif
