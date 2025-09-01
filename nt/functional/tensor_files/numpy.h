#ifndef NT_FUNCTIONAL_TENSOR_FILES_NUMPY_CONVERSION_H__
#define NT_FUNCTIONAL_TENSOR_FILES_NUMPY_CONVERSION_H__
#include "../../Tensor.h"
#include <string>

namespace nt {
namespace functional {
NEUROTENSOR_API Tensor from_numpy(std::string);
NEUROTENSOR_API void to_numpy(const Tensor &, std::string);

} // namespace functional
} // namespace nt

#endif
