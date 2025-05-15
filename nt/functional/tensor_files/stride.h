#ifndef __NT_FUNCTIONAL_TENSOR_FILES_STRIDE_H__
#define __NT_FUNCTIONAL_TENSOR_FILES_STRIDE_H__

#include "../../Tensor.h"
#include "../../dtype/Scalar.h"

namespace nt {
namespace functional {

Tensor undilate_(const Tensor&, Tensor::size_value_t);
Tensor undilate_(const Tensor&, Tensor::size_value_t, Tensor::size_value_t);
Tensor undilate_(const Tensor&, Tensor::size_value_t, Tensor::size_value_t, Tensor::size_value_t);

Tensor as_strided(const Tensor &input, const SizeRef n_size, SizeRef n_stride,
                  const int64_t storage_offset = 0, bool whole_tensor = false);


}
}

#endif
