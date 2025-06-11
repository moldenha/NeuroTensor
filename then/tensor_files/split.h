#ifndef __NT_FUNCTIONAL_TENSOR_FILES_SPLIT_H__
#define __NT_FUNCTIONAL_TENSOR_FILES_SPLIT_H__

#include "../../Tensor.h"
#include "../../dtype/Scalar.h"

namespace nt {
namespace functional {

Tensor split(Tensor input, typename Tensor::size_value_t split_size,
             int64_t dim = 0); // splits into variable number of split sizes
                               // along a given dimension
Tensor
split(Tensor input, std::vector<typename Tensor::size_value_t> split_sections,
      int64_t dim = 0); // splits into a specified amount on the given dimension
Tensor chunk(Tensor input, typename Tensor::size_value_t chunks,
             int64_t dim = 0); // splits into that many chunks
}
}

#endif
