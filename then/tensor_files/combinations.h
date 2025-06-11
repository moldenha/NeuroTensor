#ifndef __NT_FUNCTIONAL_TENSOR_FILES_COMBINATIONS_H__
#define __NT_FUNCTIONAL_TENSOR_FILES_COMBINATIONS_H__

#include "../../Tensor.h"
#include "../../dtype/Scalar.h"

namespace nt {
namespace functional {

Tensor combinations(Tensor vec, int64_t r, int64_t start = 0);

}
}

#endif
