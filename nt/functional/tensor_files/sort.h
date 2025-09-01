#ifndef NT_FUNCTIONAL_TENSOR_FILES_SORT_H__
#define NT_FUNCTIONAL_TENSOR_FILES_SORT_H__

#include "../../Tensor.h"
#include "../../dtype/Scalar.h"

namespace nt {
namespace functional {

NEUROTENSOR_API Tensor sort(const Tensor &, const Tensor::size_value_t dim = -1,
            bool descending = false, bool return_sorted = true,
            bool return_indices = true);
// this is a function designed to sort the first elements of a row or channel of
// a tensor
//  returns Tensor(sorted, indices)
//  only returning the indices or the sorted makes things faster
//  this function is meant to sort along for example each row in a matrix or
//  each collumn
NEUROTENSOR_API Tensor coordsort(const Tensor &input, Tensor::size_value_t dim = -2,
                 bool descending = false, bool return_sorted = true,
                 bool return_indices = true);
} // namespace functional
} // namespace nt

#endif
