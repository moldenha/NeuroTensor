#ifndef NT_FUNCTIONAL_TENSOR_FILES_PADDING_H__
#define NT_FUNCTIONAL_TENSOR_FILES_PADDING_H__

#include "../../Tensor.h"

namespace nt{
namespace functional{

NEUROTENSOR_API Tensor pad(const Tensor&, std::vector<Tensor::size_value_t>, const char* mode = "constant", Scalar value = 0);
NEUROTENSOR_API Tensor unpad(const Tensor&, std::vector<Tensor::size_value_t>, bool no_contiguous = false);

}
}

#endif
