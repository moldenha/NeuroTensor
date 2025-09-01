#ifndef NT_FUNCTIONAL_TENSOR_FILES_STRIDE_H__
#define NT_FUNCTIONAL_TENSOR_FILES_STRIDE_H__

#include "../../Tensor.h"
#include "../../dtype/Scalar.h"

namespace nt {
namespace functional {

NEUROTENSOR_API Tensor diagonal(const Tensor&, bool keep_dims = false);

NEUROTENSOR_API Tensor as_strided(const Tensor &input, const SizeRef n_size, SizeRef n_stride,
                  const int64_t storage_offset = 0, bool whole_tensor = false);


}
}

#endif
