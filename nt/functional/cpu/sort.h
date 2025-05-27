#ifndef __NT_FUNCTIONAL_CPU_SORT_H__
#define __NT_FUNCTIONAL_CPU_SORT_H__

#include "../../dtype/ArrayVoid.h"
#include "../../refs/SizeRef.h"

namespace nt{
namespace functional{
namespace cpu{

void _sort_vals_only_(ArrayVoid& values, const bool& descending, const int64_t& dim_size);
void _sort_vals_dtype_tensor_only_(ArrayVoid& values, const bool& descending, const int64_t& dim_size);
void _sort_(ArrayVoid& values, int64_t* indices_begin, int64_t* indices_end, const bool& decending, const int64_t& dim_size);
void _sort_tensor_(ArrayVoid& values, int64_t* indices_begin, int64_t* indices_end, const bool& decending, const int64_t& dim_size);

}
}
}

#endif
