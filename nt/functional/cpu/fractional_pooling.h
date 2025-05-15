#ifndef __NT_FUNCTIONAL_CPU_FRACTIONAL_POOLING_H__
#define __NT_FUNCTIONAL_CPU_FRACTIONAL_POOLING_H__

#include "../../dtype/ArrayVoid.h"
#include "../../refs/SizeRef.h"
#include <vector>

namespace nt{
namespace functional{
namespace cpu{

void _extract_sliding_windows_max_2d(const ArrayVoid& _input, ArrayVoid& output, 
                const std::vector<int64_t>& rows, const std::vector<int64_t>& cols, 
                int64_t batches, const SizeRef& in_shape);

void _extract_sliding_windows_max_3d(const ArrayVoid& _input, ArrayVoid& output, 
                const std::vector<int64_t>& channels, const std::vector<int64_t>& rows, const std::vector<int64_t>& cols, 
                int64_t batches, const SizeRef& in_shape);



}
}
}

#endif
