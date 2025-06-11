#ifndef __NT_FUNCTIONAL_TENSOR_FILES_UNIQUE_H__
#define __NT_FUNCTIONAL_TENSOR_FILES_UNIQUE_H__

#include "../../Tensor.h"
#include "../../dtype/Scalar.h"

namespace nt {
namespace functional {

Tensor unique(Tensor, int64_t dim, bool return_unique = true,
              bool return_indices = true);
// returns Tensor(unique, indices) (depending on return_unique and
// return_indices)

}
}

#endif
